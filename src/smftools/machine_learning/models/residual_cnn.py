"""Validated residual/dilated one-dimensional CNN model family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from smftools.optional_imports import require

torch = require("torch", extra="ml-base", purpose="residual CNN models")
nn = torch.nn
F = torch.nn.functional


class ResidualCNNConfigError(ValueError):
    """Raised when a residual CNN architecture is invalid."""


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResidualCNNConfigError(f"{path} must be a positive integer")
    return value


def _positive_integers(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ResidualCNNConfigError(f"{path} must be a sequence")
    result = tuple(_positive_integer(item, f"{path}[]") for item in value)
    if not result:
        raise ResidualCNNConfigError(f"{path} cannot be empty")
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ResidualCNNConfigError(f"{path} must be boolean")
    return value


_OPTIONAL_DEFAULTS = {"mask_channels": False, "span_masking": False, "max_receptive_field": None}


@dataclass(frozen=True)
class ResidualCNNConfig:
    """Exact architecture parameters for a residual/dilated 1D CNN."""

    in_channels: int
    stem_channels: int = 32
    block_channels: tuple[int, ...] = (64, 64, 96, 96, 128, 128)
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    stem_kernel_size: int = 9
    kernel_size: int = 5
    dropout: float = 0.15
    hidden_dim: int = 128
    output_dim: int = 1
    use_se: bool = True
    use_attention_pool: bool = True
    # `MLR-08`, all off by default (and then left out of `to_dict`, so earlier
    # architectures keep their identity):
    # - mask_channels: the validity mask joins the input as one channel per
    #   input channel, so "no site" and "site, unmodified" differ;
    # - span_masking: features are kept (and pooled) across each molecule's
    #   span -- first to last valid position -- not only where a channel is
    #   valid, so context crosses positions between sparse sites;
    # - max_receptive_field: refuse an architecture whose receptive field
    #   exceeds it (and squeeze-excite, which pools the whole molecule).
    mask_channels: bool = False
    span_masking: bool = False
    max_receptive_field: int | None = None

    @property
    def receptive_field(self) -> int:
        """Theoretical receptive field in positions: the widest input window
        one final-layer position (one detector) can see."""
        return (
            1
            + (self.stem_kernel_size - 1)
            + sum(2 * (self.kernel_size - 1) * dilation for dilation in self.dilations)
        )

    def __post_init__(self) -> None:
        for name in (
            "in_channels",
            "stem_channels",
            "stem_kernel_size",
            "kernel_size",
            "hidden_dim",
            "output_dim",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        block_channels = _positive_integers(self.block_channels, "block_channels")
        dilations = _positive_integers(self.dilations, "dilations")
        if len(block_channels) != len(dilations):
            raise ResidualCNNConfigError(
                "block_channels and dilations must contain the same number of layers"
            )
        if self.stem_kernel_size % 2 == 0 or self.kernel_size % 2 == 0:
            raise ResidualCNNConfigError(
                "stem_kernel_size and kernel_size must be odd to preserve position length"
            )
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not np.isfinite(self.dropout)
            or not 0 <= float(self.dropout) < 1
        ):
            raise ResidualCNNConfigError("dropout must be finite and in [0, 1)")
        object.__setattr__(self, "block_channels", block_channels)
        object.__setattr__(self, "dilations", dilations)
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "use_se", _boolean(self.use_se, "use_se"))
        object.__setattr__(
            self,
            "use_attention_pool",
            _boolean(self.use_attention_pool, "use_attention_pool"),
        )
        object.__setattr__(self, "mask_channels", _boolean(self.mask_channels, "mask_channels"))
        object.__setattr__(self, "span_masking", _boolean(self.span_masking, "span_masking"))
        if self.max_receptive_field is not None:
            limit = _positive_integer(self.max_receptive_field, "max_receptive_field")
            if self.receptive_field > limit:
                raise ResidualCNNConfigError(
                    f"receptive field {self.receptive_field} exceeds max_receptive_field {limit}"
                )
            if self.use_se:
                raise ResidualCNNConfigError(
                    "a bounded receptive field needs use_se=False: squeeze-excite pools "
                    "the whole molecule in every block"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return complete JSON-compatible constructor parameters."""
        payload = asdict(self)
        payload["block_channels"] = list(self.block_channels)
        payload["dilations"] = list(self.dilations)
        for name, default in _OPTIONAL_DEFAULTS.items():
            if payload[name] == default:
                payload.pop(name)  # unset: earlier architectures keep their identity
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ResidualCNNConfig:
        """Strictly validate and restore a resolved architecture."""
        expected = {
            "in_channels",
            "stem_channels",
            "block_channels",
            "dilations",
            "stem_kernel_size",
            "kernel_size",
            "dropout",
            "hidden_dim",
            "output_dim",
            "use_se",
            "use_attention_pool",
        }
        if not expected <= set(raw) <= expected | set(_OPTIONAL_DEFAULTS):
            raise ResidualCNNConfigError(
                f"residual CNN fields must be exactly {sorted(expected)} plus optional "
                f"{sorted(_OPTIONAL_DEFAULTS)}"
            )
        return cls(
            in_channels=raw["in_channels"],
            stem_channels=raw["stem_channels"],
            block_channels=raw["block_channels"],
            dilations=raw["dilations"],
            stem_kernel_size=raw["stem_kernel_size"],
            kernel_size=raw["kernel_size"],
            dropout=raw["dropout"],
            hidden_dim=raw["hidden_dim"],
            output_dim=raw["output_dim"],
            use_se=raw["use_se"],
            use_attention_pool=raw["use_attention_pool"],
            **{name: raw.get(name, default) for name, default in _OPTIONAL_DEFAULTS.items()},
        )


def residual_cnn_config_to_dict(config: ResidualCNNConfig) -> dict[str, Any]:
    """Return the canonical serialized residual CNN configuration."""
    return config.to_dict()


def residual_cnn_config_from_dict(payload: Mapping[str, Any]) -> ResidualCNNConfig:
    """Restore a canonical serialized residual CNN configuration."""
    return ResidualCNNConfig.from_dict(payload)


def default_residual_cnn_config(in_channels: int) -> ResidualCNNConfig:
    """Return the reviewed legacy-compatible residual CNN defaults."""
    return ResidualCNNConfig(in_channels=in_channels)


class SqueezeExcite1d(nn.Module):
    """Channel-wise squeeze/excitation for one-dimensional features."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, values):
        """Reweight channel features without changing their shape."""
        return values * self.net(values)


class ResidualDilatedBlock1d(nn.Module):
    """Length-preserving residual block with configurable width and dilation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_se: bool,
    ) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.se = SqueezeExcite1d(out_channels) if use_se else nn.Identity()
        self.proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, values):
        """Return one residual feature update."""
        residual = self.proj(values)
        values = F.relu(self.bn1(self.conv1(values)), inplace=True)
        values = self.drop(values)
        values = self.bn2(self.conv2(values))
        values = self.se(values)
        return F.relu(values + residual, inplace=True)


class AttentionPooling1d(nn.Module):
    """Learned pooling over valid positions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, values, *, position_mask=None):
        """Pool features, excluding false positions when a mask is supplied."""
        scores = self.score(values)
        if position_mask is not None:
            scores = scores.masked_fill(~position_mask[:, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(values * weights, dim=-1)


class ResidualDilatedCNN1d(nn.Module):
    """Plain channel-first residual CNN returning classification logits.

    Mask inputs remain separate from signal channels. ``observed_mask`` and
    observation-specific ``design_mask`` use ``(batch, channel, position)``;
    a shared design mask uses ``(channel, position)``; ``availability_mask``
    uses ``(batch, channel)``; and ``padding_mask`` uses ``(batch, position)``
    with true meaning padding. Invalid positions are excluded from every
    residual stage and all pooling operations.
    """

    def __init__(self, config: ResidualCNNConfig) -> None:
        super().__init__()
        self.config = config
        stem_padding = config.stem_kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(
                config.in_channels * (2 if config.mask_channels else 1),
                config.stem_channels,
                kernel_size=config.stem_kernel_size,
                padding=stem_padding,
            ),
            nn.BatchNorm1d(config.stem_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_channels = config.stem_channels
        for out_channels, dilation in zip(config.block_channels, config.dilations):
            blocks.append(
                ResidualDilatedBlock1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                    use_se=config.use_se,
                )
            )
            in_channels = out_channels
        self.backbone = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.attn_pool = AttentionPooling1d(in_channels) if config.use_attention_pool else None
        pooled_parts = 3 if config.use_attention_pool else 2
        self.head = nn.Sequential(
            nn.Linear(in_channels * pooled_parts, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    @property
    def attribution_layer(self):
        """Return the declared final convolutional layer for layer attribution."""
        return self.backbone[-1].conv2

    def _masked_inputs(
        self,
        values,
        *,
        observed_mask=None,
        availability_mask=None,
        design_mask=None,
        padding_mask=None,
    ):
        if values.ndim != 3:
            raise ValueError("residual CNN values must have (batch, channel, position) axes")
        batch_size, channels, positions = values.shape
        if channels != self.config.in_channels:
            raise ValueError(
                f"residual CNN expected {self.config.in_channels} channels, observed {channels}"
            )
        valid = torch.ones_like(values, dtype=torch.bool)
        if observed_mask is not None:
            observed = self._boolean_mask(observed_mask, "observed_mask", values.device)
            if observed.shape != values.shape:
                raise ValueError("observed_mask must match channel-first values")
            valid &= observed
        if availability_mask is not None:
            availability = self._boolean_mask(availability_mask, "availability_mask", values.device)
            if availability.shape != (batch_size, channels):
                raise ValueError("availability_mask must have (batch, channel) axes")
            valid &= availability[:, :, None]
        if design_mask is not None:
            design = self._boolean_mask(design_mask, "design_mask", values.device)
            if design.shape == (channels, positions):
                design = design[None, :, :]
            elif design.shape != values.shape:
                raise ValueError(
                    "design_mask must have (channel, position) or channel-first value axes"
                )
            valid &= design
        if padding_mask is not None:
            padding = self._boolean_mask(padding_mask, "padding_mask", values.device)
            if padding.shape != (batch_size, positions):
                raise ValueError("padding_mask must have (batch, position) axes")
            valid &= ~padding[:, None, :]
        position_valid = valid.any(dim=1)
        if torch.any(~position_valid.any(dim=1)):
            raise ValueError("every residual CNN observation needs at least one valid position")
        masked = values.masked_fill(~valid, 0.0)
        if not torch.isfinite(masked).all():
            raise ValueError("residual CNN values must be finite at every valid position")
        if self.config.mask_channels:
            masked = torch.cat([masked, valid.to(masked.dtype)], dim=1)
        if self.config.span_masking:
            position_valid = _span(position_valid)
        return masked, position_valid

    @staticmethod
    def _boolean_mask(value, name: str, device):
        mask = torch.as_tensor(value, device=device)
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
        return mask

    def _forward_features_and_mask(
        self,
        values,
        *,
        observed_mask=None,
        availability_mask=None,
        design_mask=None,
        padding_mask=None,
    ):
        values, position_valid = self._masked_inputs(
            values,
            observed_mask=observed_mask,
            availability_mask=availability_mask,
            design_mask=design_mask,
            padding_mask=padding_mask,
        )
        values = self.stem(values).masked_fill(~position_valid[:, None, :], 0.0)
        for block in self.backbone:
            values = block(values).masked_fill(~position_valid[:, None, :], 0.0)
        return values, position_valid

    def forward_features(
        self,
        values,
        *,
        observed_mask=None,
        availability_mask=None,
        design_mask=None,
        padding_mask=None,
    ):
        """Return masked final convolutional features."""
        features, _position_valid = self._forward_features_and_mask(
            values,
            observed_mask=observed_mask,
            availability_mask=availability_mask,
            design_mask=design_mask,
            padding_mask=padding_mask,
        )
        return features

    def forward(
        self,
        values,
        *,
        observed_mask=None,
        availability_mask=None,
        design_mask=None,
        padding_mask=None,
    ):
        """Return ``(batch, output_dim)`` logits for channel-first values."""
        features, position_valid = self._forward_features_and_mask(
            values,
            observed_mask=observed_mask,
            availability_mask=availability_mask,
            design_mask=design_mask,
            padding_mask=padding_mask,
        )
        mask = position_valid[:, None, :]
        denominator = mask.sum(dim=-1).clamp(min=1)
        average = features.sum(dim=-1) / denominator
        maximum = features.masked_fill(~mask, float("-inf")).max(dim=-1).values
        pooled = [average, maximum]
        if self.attn_pool is not None:
            pooled.append(self.attn_pool(features, position_mask=position_valid))
        return self.head(torch.cat(pooled, dim=1))


def _span(position_valid):
    """True from each row's first to last valid position (its read span)."""
    positions = torch.arange(position_valid.shape[1], device=position_valid.device)
    big = position_valid.shape[1]
    first = torch.where(position_valid, positions, big).min(dim=1).values
    last = torch.where(position_valid, positions, -1).max(dim=1).values
    return (positions[None, :] >= first[:, None]) & (positions[None, :] <= last[:, None])


def effective_span(
    model,
    values,
    *,
    observed_mask=None,
    availability_mask=None,
    design_mask=None,
    padding_mask=None,
    n_centers: int = 8,
    masses: Sequence[float] = (0.5, 0.9),
) -> dict[str, Any]:
    """How far, in practice, a final-layer detector looks (`MLR-08`).

    For ``n_centers`` positions spread over the input (kept a half receptive
    field from the ends when the input is long enough), the gradient of the
    summed final-layer features at that position with respect to the inputs,
    in absolute value and summed over molecules and channels, gives an
    influence profile by offset from the centre. The effective span at mass
    ``m`` is the narrowest centred window (2w + 1 positions) holding ``m`` of
    the total influence. Influence concentrates at a detector's centre, so it
    is often well below the theoretical ``receptive_field``.
    """
    config = model.config
    was_training = bool(model.training)
    model.eval()
    try:
        values = values.detach().clone().requires_grad_(True)
        n_positions = values.shape[-1]
        half = config.receptive_field // 2
        low, high = (
            (half, n_positions - 1 - half) if n_positions > 2 * half else (0, n_positions - 1)
        )
        centers = sorted({int(round(c)) for c in np.linspace(low, high, n_centers)})
        profile = np.zeros(2 * n_positions - 1)
        offsets = np.arange(n_positions)
        for center in centers:
            values.grad = None
            features = model.forward_features(
                values,
                observed_mask=observed_mask,
                availability_mask=availability_mask,
                design_mask=design_mask,
                padding_mask=padding_mask,
            )
            features[:, :, center].sum().backward()
            gradient = values.grad.detach().abs().sum(dim=(0, 1)).cpu().numpy()
            profile[offsets - center + n_positions - 1] += gradient
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    result: dict[str, Any] = {"receptive_field": config.receptive_field, "n_centers": len(centers)}
    total = profile.sum()
    centre = n_positions - 1
    for mass in masses:
        key = f"effective_span_{int(round(mass * 100))}"
        if total <= 0:
            result[key] = None
            continue
        width = 0
        while profile[centre - width : centre + width + 1].sum() < mass * total:
            width += 1
        result[key] = 2 * width + 1
    return result


def build_residual_cnn(config: ResidualCNNConfig):
    """Construct a plain residual CNN from a fully validated configuration."""
    return ResidualDilatedCNN1d(config)
