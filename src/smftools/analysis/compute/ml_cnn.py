"""
Matrix-level helpers for 1D CNN binary classifiers.

Inputs are plain numpy arrays and labels. No AnnData or file I/O occurs here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def detect_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class CNNConfig:
    in_channels: int
    stem_channels: int = 32
    block_channels: tuple[int, ...] = (64, 64, 96, 96, 128, 128)
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    stem_kernel_size: int = 9
    kernel_size: int = 5
    dropout: float = 0.15
    hidden_dim: int = 128
    use_se: bool = True
    use_attention_pool: bool = True


def cnn_config_to_dict(config: CNNConfig) -> dict:
    payload = asdict(config)
    payload["block_channels"] = list(config.block_channels)
    payload["dilations"] = list(config.dilations)
    return payload


def cnn_config_from_dict(payload: dict) -> CNNConfig:
    data = dict(payload)
    data["block_channels"] = tuple(data["block_channels"])
    data["dilations"] = tuple(data["dilations"])
    return CNNConfig(**data)


def default_cnn_config(in_channels: int) -> CNNConfig:
    return CNNConfig(in_channels=in_channels)


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualDilatedBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_se: bool,
    ):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.se = SqueezeExcite1d(out_channels) if use_se else nn.Identity()
        self.proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = F.relu(x + residual, inplace=True)
        return x


class AttentionPooling1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        pooled = torch.sum(x * weights, dim=-1)
        return pooled


class ResidualDilatedCNN1d(nn.Module):
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        stem_pad = config.stem_kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(config.in_channels, config.stem_channels, kernel_size=config.stem_kernel_size, padding=stem_pad),
            nn.BatchNorm1d(config.stem_channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_ch = config.stem_channels
        for out_ch, dilation in zip(config.block_channels, config.dilations):
            blocks.append(
                ResidualDilatedBlock1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                    use_se=config.use_se,
                )
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.attn_pool = AttentionPooling1d(in_ch) if config.use_attention_pool else None

        pooled_parts = 3 if config.use_attention_pool else 2
        self.head = nn.Sequential(
            nn.Linear(in_ch * pooled_parts, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        x_avg = self.gap(feats).squeeze(-1)
        x_max = self.gmp(feats).squeeze(-1)
        pooled = [x_avg, x_max]
        if self.attn_pool is not None:
            pooled.append(self.attn_pool(feats))
        x = torch.cat(pooled, dim=1)
        return self.head(x)


def build_cnn_model(config: CNNConfig) -> nn.Module:
    return ResidualDilatedCNN1d(config)


@dataclass
class TrainedCNNModel:
    model: nn.Module
    device: torch.device
    config: CNNConfig
    include_positional: bool = False
    include_spacing: bool = False
    include_design_mask: bool = False
    baseline_mode: str = "zero"


def _parse_feature_positions(feature_labels: np.ndarray | None, n_features: int) -> np.ndarray:
    if feature_labels is None:
        return np.linspace(-1.0, 1.0, n_features, dtype=np.float32)

    coords = []
    for label in np.asarray(feature_labels, dtype=object):
        try:
            coords.append(float(str(label).rsplit(":", 1)[1]))
        except Exception:
            coords.append(np.nan)
    coords = np.asarray(coords, dtype=np.float32)
    valid = np.isfinite(coords)
    if not np.any(valid):
        return np.linspace(-1.0, 1.0, n_features, dtype=np.float32)

    lo = float(np.nanmin(coords))
    hi = float(np.nanmax(coords))
    if hi <= lo:
        out = np.zeros(n_features, dtype=np.float32)
        out[~valid] = 0.0
        return out

    out = np.zeros(n_features, dtype=np.float32)
    out[valid] = 2.0 * ((coords[valid] - lo) / (hi - lo)) - 1.0
    return out


def _raw_feature_positions(feature_labels: np.ndarray | None, n_features: int) -> np.ndarray:
    if feature_labels is None:
        return np.arange(n_features, dtype=np.float32)

    coords = []
    for label in np.asarray(feature_labels, dtype=object):
        try:
            coords.append(float(str(label).rsplit(":", 1)[1]))
        except Exception:
            coords.append(np.nan)
    coords = np.asarray(coords, dtype=np.float32)
    if np.isfinite(coords).any():
        fill = np.arange(n_features, dtype=np.float32)
        coords[~np.isfinite(coords)] = fill[~np.isfinite(coords)]
        return coords
    return np.arange(n_features, dtype=np.float32)


def _spacing_channels(
    feature_labels: np.ndarray | None,
    n_features: int,
    learnable_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_pos = _raw_feature_positions(feature_labels, n_features)
    learn = (
        np.ones(n_features, dtype=bool)
        if learnable_mask is None
        else np.asarray(learnable_mask, dtype=bool)
    )
    prev_dist = np.zeros(n_features, dtype=np.float32)
    next_dist = np.zeros(n_features, dtype=np.float32)
    idx = np.flatnonzero(learn)
    if idx.size <= 1:
        return prev_dist, next_dist

    gaps = np.abs(np.diff(raw_pos[idx]).astype(np.float32))
    gaps = np.maximum(gaps, 0.0)
    prev_dist[idx[1:]] = gaps
    next_dist[idx[:-1]] = gaps
    prev_dist[idx[0]] = gaps[0]
    next_dist[idx[-1]] = gaps[-1]

    scale = float(np.max(np.log1p(gaps))) if np.isfinite(gaps).any() else 0.0
    if scale > 0:
        prev_dist = np.log1p(prev_dist) / scale
        next_dist = np.log1p(next_dist) / scale
    return prev_dist.astype(np.float32), next_dist.astype(np.float32)


def build_cnn_input(
    X: np.ndarray,
    feature_labels: np.ndarray | None = None,
    include_positional: bool = False,
    include_spacing: bool = False,
    learnable_mask: np.ndarray | None = None,
    include_design_mask: bool = False,
) -> np.ndarray:
    """
    Convert a read x feature matrix into a CNN input tensor.

    Channels
    --------
    Always included:
    - signal: methylation values with NaNs filled to 0
    - observed-mask: 1 where the original matrix was observed, 0 where missing

    Optional:
    - design-mask: 1 where the position was intentionally masked from learning
    - positional: normalized coordinate channel in [-1, 1]
    - spacing-prev / spacing-next: normalized distances to previous/next
      learnable site
    """
    X = np.asarray(X, dtype=np.float32)
    signal = np.nan_to_num(X, nan=0.0)
    observed = (~np.isnan(X)).astype(np.float32)
    channels = [signal[:, np.newaxis, :], observed[:, np.newaxis, :]]
    if learnable_mask is not None:
        learnable_mask = np.asarray(learnable_mask, dtype=bool)
        if learnable_mask.shape[0] != X.shape[1]:
            raise ValueError("learnable_mask length must match X.shape[1]")
    if include_design_mask:
        blocked = np.zeros(X.shape[1], dtype=np.float32) if learnable_mask is None else (~learnable_mask).astype(np.float32)
        if learnable_mask is not None:
            signal[:, ~learnable_mask] = 0.0
            observed[:, ~learnable_mask] = 0.0
            channels = [signal[:, np.newaxis, :], observed[:, np.newaxis, :]]
        blocked = np.broadcast_to(blocked.reshape(1, 1, -1), (X.shape[0], 1, X.shape[1])).astype(np.float32)
        channels.append(blocked)
    if include_positional:
        pos = _parse_feature_positions(feature_labels, X.shape[1])
        pos = np.broadcast_to(pos.reshape(1, 1, -1), (X.shape[0], 1, X.shape[1])).astype(np.float32)
        channels.append(pos)
    if include_spacing:
        prev_dist, next_dist = _spacing_channels(feature_labels, X.shape[1], learnable_mask)
        prev_dist = np.broadcast_to(prev_dist.reshape(1, 1, -1), (X.shape[0], 1, X.shape[1])).astype(np.float32)
        next_dist = np.broadcast_to(next_dist.reshape(1, 1, -1), (X.shape[0], 1, X.shape[1])).astype(np.float32)
        channels.extend([prev_dist, next_dist])
    return np.concatenate(channels, axis=1)


def build_cnn_baseline(
    X: np.ndarray,
    include_positional: bool = False,
    include_spacing: bool = False,
    include_design_mask: bool = False,
    baseline_mode: str = "zero",
) -> np.ndarray:
    baseline = np.zeros_like(X, dtype=np.float32)
    struct_idx = 2
    if baseline_mode == "preserve_structure" and include_design_mask and X.shape[1] > struct_idx:
        baseline[:, struct_idx, :] = X[:, struct_idx, :]
        struct_idx += 1
    if baseline_mode in {"preserve_position", "preserve_structure"} and include_positional and X.shape[1] > struct_idx:
        baseline[:, struct_idx, :] = X[:, struct_idx, :]
        struct_idx += 1
    if baseline_mode in {"preserve_position", "preserve_structure"} and include_spacing:
        if X.shape[1] > struct_idx:
            baseline[:, struct_idx, :] = X[:, struct_idx, :]
        if X.shape[1] > struct_idx + 1:
            baseline[:, struct_idx + 1, :] = X[:, struct_idx + 1, :]
    return baseline


def split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/validation split for CNN training.
    """
    if np.unique(y).size < 2 or len(y) < 8:
        return X, y, X, y
    test_size = min(max(validation_fraction, 0.1), 0.4)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_val, y_train, y_val


def fit_simple_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
    epochs: int = 40,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    patience: int = 8,
    weight_decay: float = 1e-4,
    config: CNNConfig | None = None,
    include_positional: bool = False,
    include_spacing: bool = False,
    include_design_mask: bool = False,
    baseline_mode: str = "zero",
    device: torch.device | None = None,
) -> TrainedCNNModel:
    """
    Train a residual dilated 1D CNN with early stopping on validation AUC.
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = device or detect_torch_device()
    config = config or default_cnn_config(in_channels=int(X_train.shape[1]))

    model = build_cnn_model(config).to(device)
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.05)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    X_vl = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_vl = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    best_state = None
    best_val_loss = float("inf")
    best_val_auc = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl)
            val_loss = float(criterion(val_logits, y_vl).item())
            val_probs = torch.sigmoid(val_logits).squeeze(1).detach().cpu().numpy()
        try:
            val_auc = float(roc_auc_score(y_val, val_probs))
        except Exception:
            val_auc = 0.5

        scheduler.step()

        improved = (val_auc > best_val_auc + 1e-4) or (
            abs(val_auc - best_val_auc) <= 1e-4 and val_loss < best_val_loss
        )
        if improved:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainedCNNModel(
        model=model,
        device=device,
        config=config,
        include_positional=include_positional,
        include_spacing=include_spacing,
        include_design_mask=include_design_mask,
        baseline_mode=baseline_mode,
    )


def predict_cnn_scores(trained_model: TrainedCNNModel, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """
    Return positive-class probabilities for a trained CNN model.
    """
    model = trained_model.model
    device = trained_model.device
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X_t.shape[0], batch_size):
            logits = model(X_t[start:start + batch_size])
            probs = torch.sigmoid(logits).squeeze(1)
            outs.append(probs.detach().cpu().numpy())
    return np.concatenate(outs).astype(float)
