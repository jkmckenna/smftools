from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse

from smftools.logging_utils import get_logger

logger = get_logger(__name__)
# =============================================================================
# Registry / Factory
# =============================================================================

_HMM_REGISTRY: Dict[str, type] = {}


def register_hmm(name: str):
    """Decorator to register an HMM backend under a string key."""

    def deco(cls):
        """Register the provided class in the HMM registry."""
        _HMM_REGISTRY[name] = cls
        cls.hmm_name = name
        return cls

    return deco


def create_hmm(cfg: Union[dict, Any, None], arch: Optional[str] = None, **kwargs):
    """
    Factory: creates an HMM from cfg + arch (override).
    """
    key = (
        arch
        or getattr(cfg, "hmm_arch", None)
        or (cfg.get("hmm_arch") if isinstance(cfg, dict) else None)
        or "single"
    )
    if key not in _HMM_REGISTRY:
        raise KeyError(f"Unknown hmm_arch={key!r}. Known: {sorted(_HMM_REGISTRY.keys())}")
    return _HMM_REGISTRY[key].from_config(cfg, **kwargs)


# =============================================================================
# Small utilities
# =============================================================================
def _coerce_dtype_for_device(
    dtype: torch.dtype, device: Optional[Union[str, torch.device]]
) -> torch.dtype:
    """MPS does not support float64. When targeting MPS, coerce to float32."""
    dev = torch.device(device) if isinstance(device, str) else device
    if dev is not None and getattr(dev, "type", None) == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype


def _try_json_or_literal(x: Any) -> Any:
    """Parse a string value as JSON or a Python literal when possible.

    Args:
        x: Value to parse.

    Returns:
        The parsed value if possible, otherwise the original value.
    """
    if x is None:
        return None
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return x


def _coerce_bool(x: Any) -> bool:
    """Coerce a value into a boolean using common truthy strings.

    Args:
        x: Value to coerce.

    Returns:
        Boolean interpretation of the input.
    """
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


def _resolve_dtype(dtype_entry: Any) -> torch.dtype:
    """Resolve a torch dtype from a config entry.

    Args:
        dtype_entry: Config value (string or torch.dtype).

    Returns:
        Resolved torch dtype.
    """
    if dtype_entry is None:
        return torch.float64
    if isinstance(dtype_entry, torch.dtype):
        return dtype_entry
    s = str(dtype_entry).lower()
    if "16" in s:
        return torch.float16
    if "32" in s:
        return torch.float32
    return torch.float64


def _safe_int_coords(var_names) -> Tuple[np.ndarray, bool]:
    """
    Try to cast var_names to int coordinates. If not possible,
    fall back to 0..L-1 index coordinates.
    """
    try:
        coords = np.asarray(var_names, dtype=int)
        return coords, True
    except Exception:
        return np.arange(len(var_names), dtype=int), False


def _logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute log-sum-exp in a numerically stable way.

    Args:
        x: Input tensor.
        dim: Dimension to reduce.

    Returns:
        Reduced tensor.
    """
    return torch.logsumexp(x, dim=dim)


def _ensure_layer_full_shape(adata, name: str, dtype, fill_value=0):
    """
    Ensure adata.layers[name] exists with shape (n_obs, n_vars).
    """
    if name not in adata.layers:
        arr = np.full((adata.n_obs, adata.n_vars), fill_value=fill_value, dtype=dtype)
        adata.layers[name] = arr
    else:
        arr = _to_dense_np(adata.layers[name])
        if arr.shape != (adata.n_obs, adata.n_vars):
            raise ValueError(
                f"Layer '{name}' exists but has shape {arr.shape}; expected {(adata.n_obs, adata.n_vars)}"
            )
    return adata.layers[name]


def _assign_back_obs(final_adata, sub_adata, cols: List[str]):
    """
    Assign obs columns from sub_adata back into final_adata for the matching obs_names.
    Works for list/object columns too.
    """
    idx = final_adata.obs_names.get_indexer(sub_adata.obs_names)
    if (idx < 0).any():
        raise ValueError("Some sub_adata.obs_names not found in final_adata.obs_names")

    for c in cols:
        final_adata.obs.iloc[idx, final_adata.obs.columns.get_loc(c)] = sub_adata.obs[c].values


def _to_dense_np(x):
    """Convert sparse or array-like input to a dense NumPy array.

    Args:
        x: Input array or sparse matrix.

    Returns:
        Dense NumPy array or None.
    """
    if x is None:
        return None
    if issparse(x):
        return x.toarray()
    return np.asarray(x)


def _ensure_2d_np(x):
    """Ensure an array is 2D, reshaping 1D inputs.

    Args:
        x: Input array-like.

    Returns:
        2D NumPy array.
    """
    x = _to_dense_np(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape {x.shape}")
    return x


# =============================================================================
# Feature-set normalization
# =============================================================================


def normalize_hmm_feature_sets(raw: Any) -> Dict[str, Dict[str, Any]]:
    """
    Canonical format:
      {
        "footprints": {"state": "Non-Modified", "features": {"small_bound_stretch": [0,50], ...}},
        "accessible": {"state": "Modified", "features": {"all_accessible_features": [0, inf], ...}},
        ...
      }
    Each feature range is [lo, hi) in genomic bp (or index units if coords aren't ints).
    """
    parsed = _try_json_or_literal(raw)
    if not isinstance(parsed, dict):
        return {}

    def _coerce_bound(v):
        """Coerce a bound value into a float or sentinel.

        Args:
            v: Bound value.

        Returns:
            Float, np.inf, or None.
        """
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().lower()
        if s in ("inf", "infty", "np.inf", "infinite"):
            return np.inf
        if s in ("none", ""):
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _coerce_map(feats):
        """Coerce feature ranges into (lo, hi) tuples.

        Args:
            feats: Mapping of feature names to ranges.

        Returns:
            Mapping of feature names to numeric bounds.
        """
        out = {}
        if not isinstance(feats, dict):
            return out
        for name, rng in feats.items():
            if rng is None:
                out[name] = (0.0, np.inf)
                continue
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                lo = _coerce_bound(rng[0])
                hi = _coerce_bound(rng[1])
                lo = 0.0 if lo is None else float(lo)
                hi = np.inf if hi is None else float(hi)
                out[name] = (lo, hi)
            else:
                hi = _coerce_bound(rng)
                hi = np.inf if hi is None else float(hi)
                out[name] = (0.0, hi)
        return out

    out: Dict[str, Dict[str, Any]] = {}
    for group, info in parsed.items():
        if isinstance(info, dict):
            feats = _coerce_map(info.get("features", info.get("ranges", {})))
            state = info.get("state", info.get("label", "Modified"))
        else:
            feats = _coerce_map(info)
            state = "Modified"
        out[group] = {"features": feats, "state": state}
    return out


# =============================================================================
# BaseHMM: shared decoding + annotation pipeline
# =============================================================================


class BaseHMM(nn.Module):
    """
    BaseHMM responsibilities:
      - config resolution (from_config)
      - EM fit wrapper (fit / fit_em)
      - decoding (gamma / viterbi)
      - AnnData annotation from provided arrays (X + coords)
      - save/load registry aware
    Subclasses implement:
      - _log_emission(...)  -> logB
      - optional distance-aware transition handling
    """

    def __init__(self, n_states: int = 2, eps: float = 1e-8, dtype: torch.dtype = torch.float64):
        """Initialize the base HMM with shared parameters.

        Args:
            n_states: Number of hidden states.
            eps: Smoothing epsilon for probabilities.
            dtype: Torch dtype for parameters.
        """
        super().__init__()
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.n_states = int(n_states)
        self.eps = float(eps)
        self.dtype = dtype

        # start probs + transitions (shared across backends)
        start = np.full((self.n_states,), 1.0 / self.n_states, dtype=float)
        trans = np.full((self.n_states, self.n_states), 1.0 / self.n_states, dtype=float)

        self.start = nn.Parameter(torch.tensor(start, dtype=self.dtype), requires_grad=False)
        self.trans = nn.Parameter(torch.tensor(trans, dtype=self.dtype), requires_grad=False)
        self._normalize_params()

    # ------------------------- config -------------------------

    @classmethod
    def from_config(
        cls, cfg: Union[dict, Any, None], *, override: Optional[dict] = None, device=None
    ):
        """Create a model from config with optional overrides.

        Args:
            cfg: Configuration mapping or object.
            override: Override values to apply.
            device: Device specifier.

        Returns:
            Initialized HMM instance.
        """
        merged = cls._cfg_to_dict(cfg)
        if override:
            merged.update(override)

        n_states = int(merged.get("hmm_n_states", merged.get("n_states", 2)))
        eps = float(merged.get("hmm_eps", merged.get("eps", 1e-8)))
        dtype = _resolve_dtype(merged.get("hmm_dtype", merged.get("dtype", None)))
        dtype = _coerce_dtype_for_device(dtype, device)  # <<< NEW

        model = cls(n_states=n_states, eps=eps, dtype=dtype)
        if device is not None:
            model.to(torch.device(device) if isinstance(device, str) else device)
        model._persisted_cfg = merged
        return model

    @staticmethod
    def _cfg_to_dict(cfg: Union[dict, Any, None]) -> dict:
        """Normalize a config object into a dictionary.

        Args:
            cfg: Config mapping or object.

        Returns:
            Dictionary of HMM-related config values.
        """
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return dict(cfg)
        if hasattr(cfg, "to_dict") and callable(getattr(cfg, "to_dict")):
            return dict(cfg.to_dict())
        out = {}
        for k in dir(cfg):
            if k.startswith("hmm_") or k in ("smf_modality", "cpg"):
                try:
                    out[k] = getattr(cfg, k)
                except Exception:
                    pass
        return out

    # ------------------------- params -------------------------

    def _normalize_params(self):
        """Normalize start and transition probabilities in-place."""
        with torch.no_grad():
            K = self.n_states

            # start
            self.start.data = self.start.data.reshape(-1)
            if self.start.data.numel() != K:
                self.start.data = torch.full(
                    (K,), 1.0 / K, dtype=self.dtype, device=self.start.device
                )
            self.start.data = self.start.data + self.eps
            self.start.data = self.start.data / self.start.data.sum()

            # trans
            self.trans.data = self.trans.data.reshape(K, K)
            self.trans.data = self.trans.data + self.eps
            rs = self.trans.data.sum(dim=1, keepdim=True)
            rs[rs == 0.0] = 1.0
            self.trans.data = self.trans.data / rs

    def _ensure_device_dtype(
        self, device: Optional[Union[str, torch.device]] = None
    ) -> torch.device:
        """Move parameters to the requested device/dtype.

        Args:
            device: Device specifier or None to use current device.

        Returns:
            Resolved torch device.
        """
        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device) if isinstance(device, str) else device
        self.start.data = self.start.data.to(device=device, dtype=self.dtype)
        self.trans.data = self.trans.data.to(device=device, dtype=self.dtype)
        return device

    # ------------------------- state labeling -------------------------

    def _state_modified_score(self) -> torch.Tensor:
        """Subclasses return (K,) score; higher => more “Modified/Accessible”."""
        raise NotImplementedError

    def modified_state_index(self) -> int:
        """Return the index of the most modified/accessible state."""
        scores = self._state_modified_score()
        return int(torch.argmax(scores).item())

    def resolve_target_state_index(self, state_target: Any) -> int:
        """
        Accept:
          - int -> explicit state index
          - "Modified" / "Non-Modified" and aliases
        """
        if isinstance(state_target, (int, np.integer)):
            idx = int(state_target)
            return max(0, min(idx, self.n_states - 1))

        s = str(state_target).strip().lower()
        if s in ("modified", "open", "accessible", "1", "pos", "positive"):
            return self.modified_state_index()
        if s in ("non-modified", "closed", "inaccessible", "0", "neg", "negative"):
            scores = self._state_modified_score()
            return int(torch.argmin(scores).item())
        return self.modified_state_index()

    # ------------------------- emissions -------------------------

    def _log_emission(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Return logB:
          - single: obs (N,L), mask (N,L) -> logB (N,L,K)
          - multi : obs (N,L,C), mask (N,L,C) -> logB (N,L,K)
        """
        raise NotImplementedError

    # ------------------------- decoding core -------------------------

    def _forward_backward(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        *,
        coords: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Returns gamma (N,L,K) in probability space.
        Subclasses can override for distance-aware transitions.
        """
        device = obs.device
        eps = float(self.eps)
        K = self.n_states

        logB = self._log_emission(obs, mask)  # (N,L,K)
        logA = torch.log(self.trans + eps)  # (K,K)
        logstart = torch.log(self.start + eps)  # (K,)

        N, L, _ = logB.shape

        alpha = torch.empty((N, L, K), dtype=self.dtype, device=device)
        alpha[:, 0, :] = logstart.unsqueeze(0) + logB[:, 0, :]

        for t in range(1, L):
            prev = alpha[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)  # (N,K,K)
            alpha[:, t, :] = _logsumexp(prev, dim=1) + logB[:, t, :]

        beta = torch.empty((N, L, K), dtype=self.dtype, device=device)
        beta[:, L - 1, :] = 0.0
        for t in range(L - 2, -1, -1):
            temp = logA.unsqueeze(0) + (logB[:, t + 1, :] + beta[:, t + 1, :]).unsqueeze(1)
            beta[:, t, :] = _logsumexp(temp, dim=2)

        log_gamma = alpha + beta
        logZ = _logsumexp(log_gamma, dim=2).unsqueeze(2)
        gamma = (log_gamma - logZ).exp()
        return gamma

    def _viterbi(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        *,
        coords: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Returns states (N,L) int64. Missing positions (mask False for all channels)
        are still decoded, but you’ll overwrite them to -1 during writing.
        Subclasses can override for distance-aware transitions.
        """
        device = obs.device
        eps = float(self.eps)
        K = self.n_states

        logB = self._log_emission(obs, mask)  # (N,L,K)
        logA = torch.log(self.trans + eps)  # (K,K)
        logstart = torch.log(self.start + eps)  # (K,)

        N, L, _ = logB.shape
        delta = torch.empty((N, L, K), dtype=self.dtype, device=device)
        psi = torch.empty((N, L, K), dtype=torch.long, device=device)

        delta[:, 0, :] = logstart.unsqueeze(0) + logB[:, 0, :]
        psi[:, 0, :] = -1

        for t in range(1, L):
            cand = delta[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)  # (N,K,K)
            best_val, best_idx = cand.max(dim=1)
            delta[:, t, :] = best_val + logB[:, t, :]
            psi[:, t, :] = best_idx

        last_state = torch.argmax(delta[:, L - 1, :], dim=1)  # (N,)
        states = torch.empty((N, L), dtype=torch.long, device=device)

        states[:, L - 1] = last_state
        for t in range(L - 2, -1, -1):
            states[:, t] = psi[torch.arange(N, device=device), t + 1, states[:, t + 1]]

        return states

    def decode(
        self,
        X: np.ndarray,
        coords: Optional[np.ndarray] = None,
        *,
        decode: str = "marginal",
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode observations into state calls and posterior probabilities.

        Args:
            X: Observations array (N, L) or (N, L, C).
            coords: Optional coordinates aligned to L.
            decode: Decoding strategy ("marginal" or "viterbi").
            device: Device specifier.

        Returns:
            Tuple of (states, posterior probabilities).
        """
        device = self._ensure_device_dtype(device)

        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            L = X.shape[1]
        elif X.ndim == 3:
            L = X.shape[1]
        else:
            raise ValueError(f"X must be 2D or 3D; got shape {X.shape}")

        if coords is None:
            coords = np.arange(L, dtype=int)
        coords = np.asarray(coords, dtype=int)

        if X.ndim == 2:
            obs = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=self.dtype, device=device)
            mask = torch.tensor(~np.isnan(X), dtype=torch.bool, device=device)
        else:
            obs = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=self.dtype, device=device)
            mask = torch.tensor(~np.isnan(X), dtype=torch.bool, device=device)

        gamma = self._forward_backward(obs, mask, coords=coords)

        if str(decode).lower() == "viterbi":
            st = self._viterbi(obs, mask, coords=coords)
        else:
            st = torch.argmax(gamma, dim=2)

        return st.detach().cpu().numpy(), gamma.detach().cpu().numpy()

    # ------------------------- EM fit -------------------------

    def fit(
        self,
        X: np.ndarray,
        coords: Optional[np.ndarray] = None,
        *,
        max_iter: int = 50,
        tol: float = 1e-4,
        device: Optional[Union[str, torch.device]] = None,
        update_start: bool = True,
        update_trans: bool = True,
        update_emission: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> List[float]:
        """Fit HMM parameters using EM.

        Args:
            X: Observations array.
            coords: Optional coordinate array.
            max_iter: Maximum EM iterations.
            tol: Convergence tolerance.
            device: Device specifier.
            update_start: Whether to update start probabilities.
            update_trans: Whether to update transition probabilities.
            update_emission: Whether to update emission parameters.
            verbose: Whether to log progress.
            **kwargs: Additional implementation-specific kwargs.

        Returns:
            List of log-likelihood values across iterations.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim not in (2, 3):
            raise ValueError(f"X must be 2D or 3D; got {X.shape}")
        L = X.shape[1]

        if coords is None:
            coords = np.arange(L, dtype=int)
        coords = np.asarray(coords, dtype=int)

        device = self._ensure_device_dtype(device)
        return self.fit_em(
            X,
            coords,
            device=device,
            max_iter=max_iter,
            tol=tol,
            update_start=update_start,
            update_trans=update_trans,
            update_emission=update_emission,
            verbose=verbose,
            **kwargs,
        )

    def adapt_emissions(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        *,
        iters: int = 5,
        device: Optional[Union[str, torch.device]] = None,
        freeze_start: bool = True,
        freeze_trans: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> List[float]:
        """Adapt emission parameters while keeping shared structure fixed.

        Args:
            X: Observations array.
            coords: Coordinate array aligned to X.
            iters: Number of EM iterations.
            device: Device specifier.
            freeze_start: Whether to freeze start probabilities.
            freeze_trans: Whether to freeze transitions.
            verbose: Whether to log progress.
            **kwargs: Additional implementation-specific kwargs.

        Returns:
            List of log-likelihood values across iterations.
        """
        return self.fit(
            X,
            coords,
            max_iter=int(iters),
            tol=0.0,
            device=device,
            update_start=not freeze_start,
            update_trans=not freeze_trans,
            update_emission=True,
            verbose=verbose,
            **kwargs,
        )

    def fit_em(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        *,
        device: torch.device,
        max_iter: int,
        tol: float,
        update_start: bool,
        update_trans: bool,
        update_emission: bool,
        verbose: bool,
        **kwargs,
    ) -> List[float]:
        """Run the core EM update loop (subclasses implement).

        Args:
            X: Observations array.
            coords: Coordinate array aligned to X.
            device: Torch device.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            update_start: Whether to update start probabilities.
            update_trans: Whether to update transitions.
            update_emission: Whether to update emission parameters.
            verbose: Whether to log progress.
            **kwargs: Additional subclass-specific kwargs.

        Returns:
            List of log-likelihood values across iterations.
        """
        raise NotImplementedError

    # ------------------------- save/load -------------------------

    def _extra_save_payload(self) -> dict:
        """Return extra model state to include when saving."""
        return {}

    def _load_extra_payload(self, payload: dict, *, device: torch.device):
        """Load extra model state saved by subclasses.

        Args:
            payload: Serialized model payload.
            device: Torch device for tensors.
        """
        return

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the model to disk.

        Args:
            path: Output path for the serialized model.
        """
        path = str(path)
        payload = {
            "hmm_name": getattr(self, "hmm_name", self.__class__.__name__),
            "class": self.__class__.__name__,
            "n_states": int(self.n_states),
            "eps": float(self.eps),
            "dtype": str(self.dtype),
            "start": self.start.detach().cpu(),
            "trans": self.trans.detach().cpu(),
        }
        payload.update(self._extra_save_payload())
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[Union[str, torch.device]] = None):
        """Load a serialized model from disk.

        Args:
            path: Path to the serialized model.
            device: Optional device specifier.

        Returns:
            Loaded HMM instance.
        """
        payload = torch.load(str(path), map_location="cpu")
        hmm_name = payload.get("hmm_name", None)
        klass = _HMM_REGISTRY.get(hmm_name, cls)

        dtype_str = str(payload.get("dtype", "torch.float64"))
        torch_dtype = getattr(torch, dtype_str.split(".")[-1], torch.float64)
        torch_dtype = _coerce_dtype_for_device(torch_dtype, device)  # <<< NEW

        model = klass(
            n_states=int(payload["n_states"]),
            eps=float(payload.get("eps", 1e-8)),
            dtype=torch_dtype,
        )
        dev = torch.device(device) if isinstance(device, str) else (device or torch.device("cpu"))
        model.to(dev)

        with torch.no_grad():
            model.start.data = payload["start"].to(device=dev, dtype=model.dtype)
            model.trans.data = payload["trans"].to(device=dev, dtype=model.dtype)

        model._load_extra_payload(payload, device=dev)
        model._normalize_params()
        return model

    # ------------------------- interval helpers -------------------------

    @staticmethod
    def _runs_from_bool(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
        """
        Return runs as (start_idx, end_idx_exclusive) for True segments.
        """
        idx = np.nonzero(mask_1d)[0]
        if idx.size == 0:
            return []
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks] + 1, idx[-1] + 1]
        return list(zip(starts, ends))

    @staticmethod
    def _interval_length(coords: np.ndarray, s: int, e: int) -> int:
        """Genomic length for [s,e) on coords."""
        if e <= s:
            return 0
        return int(coords[e - 1]) - int(coords[s]) + 1

    @staticmethod
    def _write_lengths_for_binary_layer(bin_mat: np.ndarray) -> np.ndarray:
        """
        For each row, each True-run gets its run-length assigned across that run.
        Output same shape as bin_mat, int32.
        """
        n, L = bin_mat.shape
        out = np.zeros((n, L), dtype=np.int32)
        for i in range(n):
            runs = BaseHMM._runs_from_bool(bin_mat[i].astype(bool))
            for s, e in runs:
                out[i, s:e] = e - s
        return out

    @staticmethod
    def _write_lengths_for_state_layer(states: np.ndarray) -> np.ndarray:
        """
        For each row, each constant-state run gets run-length assigned across run.
        Missing values should be -1 and will get 0 length.
        """
        n, L = states.shape
        out = np.zeros((n, L), dtype=np.int32)
        for i in range(n):
            row = states[i]
            valid = row >= 0
            if not np.any(valid):
                continue
            # scan runs
            s = 0
            while s < L:
                if row[s] < 0:
                    s += 1
                    continue
                v = row[s]
                e = s + 1
                while e < L and row[e] == v:
                    e += 1
                out[i, s:e] = e - s
                s = e
        return out

    # ------------------------- merging -------------------------

    def merge_intervals_to_new_layer(
        self,
        adata,
        base_layer: str,
        *,
        distance_threshold: int,
        suffix: str = "_merged",
        overwrite: bool = True,
    ) -> str:
        """
        Merge adjacent 1-intervals in a binary layer if gaps <= distance_threshold (in coords space),
        writing:
          - {base_layer}{suffix}
          - {base_layer}{suffix}_lengths   (run-length in index units)
        """
        if base_layer not in adata.layers:
            raise KeyError(f"Layer '{base_layer}' not found.")

        coords, coords_are_ints = _safe_int_coords(adata.var_names)
        arr = np.asarray(adata.layers[base_layer])
        arr = (arr > 0).astype(np.uint8)

        merged_name = f"{base_layer}{suffix}"
        merged_len_name = f"{merged_name}_lengths"

        if (merged_name in adata.layers or merged_len_name in adata.layers) and not overwrite:
            raise KeyError(f"Merged outputs exist (use overwrite=True): {merged_name}")

        n, L = arr.shape
        out = np.zeros_like(arr, dtype=np.uint8)

        dt = int(distance_threshold)

        for i in range(n):
            ones = np.nonzero(arr[i] != 0)[0]
            runs = self._runs_from_bool(arr[i] != 0)
            if not runs:
                continue
            ms, me = runs[0]
            merged_runs = []
            for s, e in runs[1:]:
                if coords_are_ints:
                    gap = int(coords[s]) - int(coords[me - 1]) - 1
                else:
                    gap = s - me
                if gap <= dt:
                    me = e
                else:
                    merged_runs.append((ms, me))
                    ms, me = s, e
            merged_runs.append((ms, me))

            for s, e in merged_runs:
                out[i, s:e] = 1

        adata.layers[merged_name] = out
        adata.layers[merged_len_name] = self._write_lengths_for_binary_layer(out)

        # bookkeeping
        key = "hmm_appended_layers"
        if adata.uns.get(key) is None:
            adata.uns[key] = []
        for nm in (merged_name, merged_len_name):
            if nm not in adata.uns[key]:
                adata.uns[key].append(nm)

        return merged_name

    def write_size_class_layers_from_binary(
        self,
        adata,
        base_layer: str,
        *,
        out_prefix: str,
        feature_ranges: Dict[str, Tuple[float, float]],
        suffix: str = "",
        overwrite: bool = True,
    ) -> List[str]:
        """
        Take an existing binary layer (runs represent features) and write size-class layers:
          - {out_prefix}_{feature}{suffix}
          - plus lengths layers

        feature_ranges: name -> (lo, hi) in genomic bp.
        """
        if base_layer not in adata.layers:
            raise KeyError(f"Layer '{base_layer}' not found.")

        coords, coords_are_ints = _safe_int_coords(adata.var_names)
        bin_arr = (np.asarray(adata.layers[base_layer]) > 0).astype(np.uint8)
        n, L = bin_arr.shape

        created: List[str] = []
        for feat_name in feature_ranges.keys():
            nm = f"{out_prefix}_{feat_name}{suffix}"
            ln = f"{nm}_lengths"
            if (nm in adata.layers or ln in adata.layers) and not overwrite:
                continue
            adata.layers[nm] = np.zeros((n, L), dtype=np.uint8)
            adata.layers[ln] = np.zeros((n, L), dtype=np.int32)
            created.extend([nm, ln])

        for i in range(n):
            runs = self._runs_from_bool(bin_arr[i] != 0)
            for s, e in runs:
                length_bp = self._interval_length(coords, s, e) if coords_are_ints else (e - s)
                for feat_name, (lo, hi) in feature_ranges.items():
                    if float(lo) <= float(length_bp) < float(hi):
                        nm = f"{out_prefix}_{feat_name}{suffix}"
                        adata.layers[nm][i, s:e] = 1
                        adata.layers[f"{nm}_lengths"][i, s:e] = e - s
                        break

        # fill lengths for each size layer (consistent, even if overlaps)
        for feat_name in feature_ranges.keys():
            nm = f"{out_prefix}_{feat_name}{suffix}"
            adata.layers[f"{nm}_lengths"] = self._write_lengths_for_binary_layer(
                np.asarray(adata.layers[nm])
            )

        key = "hmm_appended_layers"
        if adata.uns.get(key) is None:
            adata.uns[key] = []
        for nm in created:
            if nm not in adata.uns[key]:
                adata.uns[key].append(nm)

        return created

    # ------------------------- AnnData annotation -------------------------

    @staticmethod
    def _resolve_pos_mask_for_methbase(subset, ref: str, methbase: str) -> Optional[np.ndarray]:
        """
        Local helper to resolve per-base masks from subset.var.* columns.
        Returns a boolean np.ndarray of length subset.n_vars or None.
        """
        key = str(methbase).strip().lower()
        var = subset.var

        def _has(col: str) -> bool:
            """Return True when a column exists on subset.var."""
            return col in var.columns

        if key in ("a",):
            col = f"{ref}_strand_FASTA_base"
            if not _has(col):
                return None
            return np.asarray(var[col] == "A")

        if key in ("c", "any_c", "anyc", "any-c"):
            for col in (f"{ref}_any_C_site", f"{ref}_C_site"):
                if _has(col):
                    return np.asarray(var[col])
            return None

        if key in ("gpc", "gpc_site", "gpc-site"):
            col = f"{ref}_GpC_site"
            if not _has(col):
                return None
            return np.asarray(var[col])

        if key in ("cpg", "cpg_site", "cpg-site"):
            col = f"{ref}_CpG_site"
            if not _has(col):
                return None
            return np.asarray(var[col])

        alt = f"{ref}_{methbase}_site"
        if not _has(alt):
            return None
        return np.asarray(var[alt])

    def annotate_adata(
        self,
        adata,
        *,
        prefix: str,
        X: np.ndarray,
        coords: np.ndarray,
        var_mask: np.ndarray,
        span_fill: bool = True,
        config=None,
        decode: str = "marginal",
        write_posterior: bool = True,
        posterior_state: str = "Modified",
        feature_sets: Optional[Dict[str, Dict[str, Any]]] = None,
        prob_threshold: float = 0.5,
        uns_key: str = "hmm_appended_layers",
        uns_flag: str = "hmm_annotated",
        force_redo: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """Decode and annotate an AnnData object with HMM-derived layers.

        Args:
            adata: AnnData to annotate.
            prefix: Prefix for newly written layers.
            X: Observations array for decoding.
            coords: Coordinate array aligned to X.
            var_mask: Boolean mask for positions in adata.var.
            span_fill: Whether to fill missing spans.
            config: Optional config for naming and state selection.
            decode: Decode method ("marginal" or "viterbi").
            write_posterior: Whether to write posterior probabilities.
            posterior_state: State label to write posterior for.
            feature_sets: Optional feature set definition for size classes.
            prob_threshold: Posterior probability threshold for binary calls.
            uns_key: .uns key to track appended layers.
            uns_flag: .uns flag to mark annotations.
            force_redo: Whether to overwrite existing layers.
            device: Device specifier.
            **kwargs: Additional parameters for specialized workflows.

        Returns:
            List of created layer names or None if skipped.
        """
        # skip logic
        if bool(adata.uns.get(uns_flag, False)) and not force_redo:
            return None

        if adata.uns.get(uns_key) is None:
            adata.uns[uns_key] = []
        appended = list(adata.uns.get(uns_key, [])) if adata.uns.get(uns_key) is not None else []

        X = np.asarray(X, dtype=float)
        coords = np.asarray(coords, dtype=int)
        var_mask = np.asarray(var_mask, dtype=bool)
        if var_mask.shape[0] != adata.n_vars:
            raise ValueError(f"var_mask length {var_mask.shape[0]} != adata.n_vars {adata.n_vars}")

        # decode
        states, gamma = self.decode(
            X, coords, decode=decode, device=device
        )  # states (N,L), gamma (N,L,K)
        N, L = states.shape
        if N != adata.n_obs:
            raise ValueError(f"X has N={N} rows but adata.n_obs={adata.n_obs}")

        # map coords -> full-var indices for span_fill
        full_coords, full_int = _safe_int_coords(adata.var_names)

        # ---- write posterior + states on masked columns only ----
        masked_idx = np.nonzero(var_mask)[0]
        masked_coords, _ = _safe_int_coords(adata.var_names[var_mask])

        # build mapping from coords order -> masked column order
        coord_to_pos_in_decoded = {int(c): i for i, c in enumerate(coords.tolist())}
        take = np.array(
            [coord_to_pos_in_decoded.get(int(c), -1) for c in masked_coords.tolist()], dtype=int
        )
        good = take >= 0
        masked_idx = masked_idx[good]
        take = take[good]

        # states layer
        states_name = f"{prefix}_states"
        if states_name not in adata.layers:
            adata.layers[states_name] = np.full((adata.n_obs, adata.n_vars), -1, dtype=np.int8)
        adata.layers[states_name][:, masked_idx] = states[:, take].astype(np.int8)
        if states_name not in appended:
            appended.append(states_name)

        # posterior layer (requested state)
        if write_posterior:
            t_idx = self.resolve_target_state_index(posterior_state)
            post = gamma[:, :, t_idx].astype(np.float32)
            post_name = f"{prefix}_posterior_{str(posterior_state).strip().lower().replace(' ', '_').replace('-', '_')}"
            if post_name not in adata.layers:
                adata.layers[post_name] = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)
            adata.layers[post_name][:, masked_idx] = post[:, take]
            if post_name not in appended:
                appended.append(post_name)

        # ---- feature layers ----
        if feature_sets is None:
            cfgd = self._cfg_to_dict(config)
            feature_sets = normalize_hmm_feature_sets(cfgd.get("hmm_feature_sets", None))

        if not feature_sets:
            adata.uns[uns_key] = appended
            adata.uns[uns_flag] = True
            return None

        # allocate outputs
        for group, fs in feature_sets.items():
            fmap = fs.get("features", {}) or {}
            if not fmap:
                continue

            all_layer = f"{prefix}_all_{group}_features"
            if all_layer not in adata.layers:
                adata.layers[all_layer] = np.zeros((adata.n_obs, adata.n_vars), dtype=np.uint8)
            if f"{all_layer}_lengths" not in adata.layers:
                adata.layers[f"{all_layer}_lengths"] = np.zeros(
                    (adata.n_obs, adata.n_vars), dtype=np.int32
                )
            for nm in (all_layer, f"{all_layer}_lengths"):
                if nm not in appended:
                    appended.append(nm)

            for feat in fmap.keys():
                nm = f"{prefix}_{feat}"
                if nm not in adata.layers:
                    adata.layers[nm] = np.zeros(
                        (adata.n_obs, adata.n_vars),
                        dtype=np.int32 if nm.endswith("_lengths") else np.uint8,
                    )
                if f"{nm}_lengths" not in adata.layers:
                    adata.layers[f"{nm}_lengths"] = np.zeros(
                        (adata.n_obs, adata.n_vars), dtype=np.int32
                    )
                for outnm in (nm, f"{nm}_lengths"):
                    if outnm not in appended:
                        appended.append(outnm)

            # classify runs per row
            target_idx = self.resolve_target_state_index(fs.get("state", "Modified"))
            membership = (
                (states == target_idx)
                if str(decode).lower() == "viterbi"
                else (gamma[:, :, target_idx] >= float(prob_threshold))
            )

            for i in range(N):
                runs = self._runs_from_bool(membership[i].astype(bool))
                for s, e in runs:
                    # genomic length in coords space
                    glen = int(coords[e - 1]) - int(coords[s]) + 1 if e > s else 0
                    if glen <= 0:
                        continue

                    # pick feature bin
                    chosen = None
                    for feat_name, (lo, hi) in fmap.items():
                        if float(lo) <= float(glen) < float(hi):
                            chosen = feat_name
                            break
                    if chosen is None:
                        continue

                    # convert span to indices in full var grid
                    if span_fill and full_int:
                        left = int(np.searchsorted(full_coords, int(coords[s]), side="left"))
                        right = int(np.searchsorted(full_coords, int(coords[e - 1]), side="right"))
                        if left >= right:
                            continue
                        adata.layers[f"{prefix}_{chosen}"][i, left:right] = 1
                        adata.layers[f"{prefix}_all_{group}_features"][i, left:right] = 1
                    else:
                        # only fill at masked indices
                        cols = masked_idx[
                            (masked_coords >= coords[s]) & (masked_coords <= coords[e - 1])
                        ]
                        if cols.size == 0:
                            continue
                        adata.layers[f"{prefix}_{chosen}"][i, cols] = 1
                        adata.layers[f"{prefix}_all_{group}_features"][i, cols] = 1

            # lengths derived from binary
            adata.layers[f"{prefix}_all_{group}_features_lengths"] = (
                self._write_lengths_for_binary_layer(
                    np.asarray(adata.layers[f"{prefix}_all_{group}_features"])
                )
            )
            for feat in fmap.keys():
                nm = f"{prefix}_{feat}"
                adata.layers[f"{nm}_lengths"] = self._write_lengths_for_binary_layer(
                    np.asarray(adata.layers[nm])
                )

        adata.uns[uns_key] = appended
        adata.uns[uns_flag] = True
        return None

    # ------------------------- row-copy helper (workflow uses it) -------------------------

    def _ensure_final_layer_and_assign(
        self, final_adata, layer_name: str, subset_idx_mask: np.ndarray, sub_data
    ):
        """
        Assign rows from sub_data into final_adata.layers[layer_name] for rows where subset_idx_mask is True.
        Handles dense arrays. If you want sparse support, add it here.
        """
        n_final_obs, n_vars = final_adata.shape
        final_rows = np.nonzero(np.asarray(subset_idx_mask).astype(bool))[0]
        sub_arr = np.asarray(sub_data)

        if layer_name not in final_adata.layers:
            final_adata.layers[layer_name] = np.zeros((n_final_obs, n_vars), dtype=sub_arr.dtype)

        final_arr = np.asarray(final_adata.layers[layer_name])
        if sub_arr.shape[0] != final_rows.size:
            raise ValueError(f"Sub rows {sub_arr.shape[0]} != mask sum {final_rows.size}")
        final_arr[final_rows, :] = sub_arr
        final_adata.layers[layer_name] = final_arr


# =============================================================================
# Single-channel Bernoulli HMM
# =============================================================================


@register_hmm("single")
class SingleBernoulliHMM(BaseHMM):
    """
    Bernoulli emission per state:
      emission[k] = P(obs==1 | state=k)
    """

    def __init__(
        self,
        n_states: int = 2,
        init_emission: Optional[Sequence[float]] = None,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize a single-channel Bernoulli HMM.

        Args:
            n_states: Number of hidden states.
            init_emission: Initial emission probabilities per state.
            eps: Smoothing epsilon for probabilities.
            dtype: Torch dtype for parameters.
        """
        super().__init__(n_states=n_states, eps=eps, dtype=dtype)
        if init_emission is None:
            em = np.full((self.n_states,), 0.5, dtype=float)
        else:
            em = np.asarray(init_emission, dtype=float).reshape(-1)[: self.n_states]
            if em.size != self.n_states:
                em = np.full((self.n_states,), 0.5, dtype=float)

        self.emission = nn.Parameter(torch.tensor(em, dtype=self.dtype), requires_grad=False)
        self._normalize_emission()

    @classmethod
    def from_config(cls, cfg, *, override=None, device=None):
        """Create a single-channel Bernoulli HMM from config.

        Args:
            cfg: Configuration mapping or object.
            override: Override values to apply.
            device: Optional device specifier.

        Returns:
            Initialized SingleBernoulliHMM instance.
        """
        merged = cls._cfg_to_dict(cfg)
        if override:
            merged.update(override)
        n_states = int(merged.get("hmm_n_states", 2))
        eps = float(merged.get("hmm_eps", 1e-8))
        dtype = _resolve_dtype(merged.get("hmm_dtype", None))
        dtype = _coerce_dtype_for_device(dtype, device)  # <<< NEW
        init_em = merged.get("hmm_init_emission_probs", merged.get("hmm_init_emission", None))
        model = cls(n_states=n_states, init_emission=init_em, eps=eps, dtype=dtype)
        if device is not None:
            model.to(torch.device(device) if isinstance(device, str) else device)
        model._persisted_cfg = merged
        return model

    def _normalize_emission(self):
        """Normalize and clamp emission probabilities in-place."""
        with torch.no_grad():
            self.emission.data = self.emission.data.reshape(-1)
            if self.emission.data.numel() != self.n_states:
                self.emission.data = torch.full(
                    (self.n_states,), 0.5, dtype=self.dtype, device=self.emission.device
                )
            self.emission.data = self.emission.data.clamp(min=self.eps, max=1.0 - self.eps)

    def _ensure_device_dtype(self, device=None) -> torch.device:
        """Move emission parameters to the requested device/dtype."""
        device = super()._ensure_device_dtype(device)
        self.emission.data = self.emission.data.to(device=device, dtype=self.dtype)
        return device

    def _state_modified_score(self) -> torch.Tensor:
        """Return per-state modified scores for ranking."""
        return self.emission.detach()

    def _log_emission(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        obs: (N,L), mask: (N,L) -> logB: (N,L,K)
        """
        p = self.emission  # (K,)
        logp = torch.log(p + self.eps)
        log1mp = torch.log1p(-p + self.eps)

        o = obs.unsqueeze(-1)  # (N,L,1)
        logB = o * logp.view(1, 1, -1) + (1.0 - o) * log1mp.view(1, 1, -1)
        logB = torch.where(mask.unsqueeze(-1), logB, torch.zeros_like(logB))
        return logB

    def _extra_save_payload(self) -> dict:
        """Return extra payload data for serialization."""
        return {"emission": self.emission.detach().cpu()}

    def _load_extra_payload(self, payload: dict, *, device: torch.device):
        """Load serialized emission parameters.

        Args:
            payload: Serialized payload dictionary.
            device: Target torch device.
        """
        with torch.no_grad():
            self.emission.data = payload["emission"].to(device=device, dtype=self.dtype)
        self._normalize_emission()

    def fit_em(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        *,
        device: torch.device,
        max_iter: int,
        tol: float,
        update_start: bool,
        update_trans: bool,
        update_emission: bool,
        verbose: bool,
        **kwargs,
    ) -> List[float]:
        """Run EM updates for a single-channel Bernoulli HMM.

        Args:
            X: Observations array (N, L).
            coords: Coordinate array aligned to X.
            device: Torch device.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            update_start: Whether to update start probabilities.
            update_trans: Whether to update transitions.
            update_emission: Whether to update emission parameters.
            verbose: Whether to log progress.
            **kwargs: Additional implementation-specific kwargs.

        Returns:
            List of log-likelihood proxy values.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("SingleBernoulliHMM expects X shape (N,L).")
        obs = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=self.dtype, device=device)
        mask = torch.tensor(~np.isnan(X), dtype=torch.bool, device=device)

        eps = float(self.eps)
        K = self.n_states
        N, L = obs.shape

        hist: List[float] = []
        for it in range(1, int(max_iter) + 1):
            gamma = self._forward_backward(obs, mask)  # (N,L,K)

            # log-likelihood proxy
            ll_proxy = float(torch.sum(torch.log(torch.clamp(gamma.sum(dim=2), min=eps))).item())
            hist.append(ll_proxy)

            # expected start
            start_acc = gamma[:, 0, :].sum(dim=0)  # (K,)

            # expected transitions xi
            logB = self._log_emission(obs, mask)
            logA = torch.log(self.trans + eps)
            alpha = torch.empty((N, L, K), dtype=self.dtype, device=device)
            alpha[:, 0, :] = torch.log(self.start + eps).unsqueeze(0) + logB[:, 0, :]
            for t in range(1, L):
                prev = alpha[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)
                alpha[:, t, :] = _logsumexp(prev, dim=1) + logB[:, t, :]
            beta = torch.empty((N, L, K), dtype=self.dtype, device=device)
            beta[:, L - 1, :] = 0.0
            for t in range(L - 2, -1, -1):
                temp = logA.unsqueeze(0) + (logB[:, t + 1, :] + beta[:, t + 1, :]).unsqueeze(1)
                beta[:, t, :] = _logsumexp(temp, dim=2)

            trans_acc = torch.zeros((K, K), dtype=self.dtype, device=device)
            for t in range(L - 1):
                valid_t = (mask[:, t] & mask[:, t + 1]).float().view(N, 1, 1)
                log_xi = (
                    alpha[:, t, :].unsqueeze(2)
                    + logA.unsqueeze(0)
                    + (logB[:, t + 1, :] + beta[:, t + 1, :]).unsqueeze(1)
                )
                log_norm = _logsumexp(log_xi.view(N, -1), dim=1).view(N, 1, 1)
                xi = (log_xi - log_norm).exp() * valid_t
                trans_acc += xi.sum(dim=0)

            # emission update
            mask_f = mask.float().unsqueeze(-1)  # (N,L,1)
            emit_num = (gamma * obs.unsqueeze(-1) * mask_f).sum(dim=(0, 1))  # (K,)
            emit_den = (gamma * mask_f).sum(dim=(0, 1))  # (K,)

            with torch.no_grad():
                if update_start:
                    new_start = start_acc + eps
                    self.start.data = new_start / new_start.sum()

                if update_trans:
                    new_trans = trans_acc + eps
                    rs = new_trans.sum(dim=1, keepdim=True)
                    rs[rs == 0.0] = 1.0
                    self.trans.data = new_trans / rs

                if update_emission:
                    new_em = (emit_num + eps) / (emit_den + 2.0 * eps)
                    self.emission.data = new_em.clamp(min=eps, max=1.0 - eps)

            self._normalize_params()
            self._normalize_emission()

            if verbose:
                logger.info(
                    "[SingleBernoulliHMM.fit] iter=%s ll_proxy=%.6f",
                    it,
                    hist[-1],
                )

            if len(hist) > 1 and abs(hist[-1] - hist[-2]) < float(tol):
                break

        return hist

    def adapt_emissions(
        self,
        X: np.ndarray,
        coords: Optional[np.ndarray] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        iters: Optional[int] = None,
        max_iter: Optional[int] = None,  # alias for your trainer
        verbose: bool = False,
        **kwargs,
    ):
        """Adapt emissions with legacy parameter names.

        Args:
            X: Observations array.
            coords: Optional coordinate array.
            device: Device specifier.
            iters: Number of iterations.
            max_iter: Alias for iters.
            verbose: Whether to log progress.
            **kwargs: Additional kwargs forwarded to BaseHMM.adapt_emissions.

        Returns:
            List of log-likelihood values.
        """
        if iters is None:
            iters = int(max_iter) if max_iter is not None else int(kwargs.pop("iters", 5))
        return super().adapt_emissions(
            np.asarray(X, dtype=float),
            coords if coords is not None else None,
            iters=int(iters),
            device=device,
            verbose=verbose,
        )


# =============================================================================
# Multi-channel Bernoulli HMM (union coordinate grid)
# =============================================================================


@register_hmm("multi")
class MultiBernoulliHMM(BaseHMM):
    """
    Multi-channel independent Bernoulli:
      emission[k,c] = P(obs_c==1 | state=k)
    X must be (N,L,C) on a union coordinate grid; NaN per-channel allowed.
    """

    def __init__(
        self,
        n_states: int = 2,
        n_channels: int = 2,
        init_emission: Optional[Any] = None,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize a multi-channel Bernoulli HMM.

        Args:
            n_states: Number of hidden states.
            n_channels: Number of observed channels.
            init_emission: Initial emission probabilities.
            eps: Smoothing epsilon for probabilities.
            dtype: Torch dtype for parameters.
        """
        super().__init__(n_states=n_states, eps=eps, dtype=dtype)
        self.n_channels = int(n_channels)
        if self.n_channels < 1:
            raise ValueError("n_channels must be >=1")

        if init_emission is None:
            em = np.full((self.n_states, self.n_channels), 0.5, dtype=float)
        else:
            arr = np.asarray(init_emission, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
                em = np.repeat(arr[: self.n_states, :], self.n_channels, axis=1)
            else:
                em = arr[: self.n_states, : self.n_channels]
            if em.shape != (self.n_states, self.n_channels):
                em = np.full((self.n_states, self.n_channels), 0.5, dtype=float)

        self.emission = nn.Parameter(torch.tensor(em, dtype=self.dtype), requires_grad=False)
        self._normalize_emission()

    @classmethod
    def from_config(cls, cfg, *, override=None, device=None):
        """Create a multi-channel Bernoulli HMM from config.

        Args:
            cfg: Configuration mapping or object.
            override: Override values to apply.
            device: Optional device specifier.

        Returns:
            Initialized MultiBernoulliHMM instance.
        """
        merged = cls._cfg_to_dict(cfg)
        if override:
            merged.update(override)
        n_states = int(merged.get("hmm_n_states", 2))
        eps = float(merged.get("hmm_eps", 1e-8))
        dtype = _resolve_dtype(merged.get("hmm_dtype", None))
        dtype = _coerce_dtype_for_device(dtype, device)  # <<< NEW
        n_channels = int(merged.get("hmm_n_channels", merged.get("n_channels", 2)))
        init_em = merged.get("hmm_init_emission_probs", None)
        model = cls(
            n_states=n_states, n_channels=n_channels, init_emission=init_em, eps=eps, dtype=dtype
        )
        if device is not None:
            model.to(torch.device(device) if isinstance(device, str) else device)
        model._persisted_cfg = merged
        return model

    def _normalize_emission(self):
        """Normalize and clamp emission probabilities in-place."""
        with torch.no_grad():
            self.emission.data = self.emission.data.reshape(self.n_states, self.n_channels)
            self.emission.data = self.emission.data.clamp(min=self.eps, max=1.0 - self.eps)

    def _ensure_device_dtype(self, device=None) -> torch.device:
        """Move emission parameters to the requested device/dtype."""
        device = super()._ensure_device_dtype(device)
        self.emission.data = self.emission.data.to(device=device, dtype=self.dtype)
        return device

    def _state_modified_score(self) -> torch.Tensor:
        """Return per-state modified scores for ranking."""
        # more “modified” = higher mean P(1) across channels
        return self.emission.detach().mean(dim=1)

    def _log_emission(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        obs: (N,L,C), mask: (N,L,C) -> logB: (N,L,K)
        """
        N, L, C = obs.shape
        K = self.n_states

        p = self.emission  # (K,C)
        logp = torch.log(p + self.eps).view(1, 1, K, C)
        log1mp = torch.log1p(-p + self.eps).view(1, 1, K, C)

        o = obs.unsqueeze(2)  # (N,L,1,C)
        m = mask.unsqueeze(2)  # (N,L,1,C)

        logBC = o * logp + (1.0 - o) * log1mp
        logBC = torch.where(m, logBC, torch.zeros_like(logBC))
        return logBC.sum(dim=3)  # sum channels -> (N,L,K)

    def _extra_save_payload(self) -> dict:
        """Return extra payload data for serialization."""
        return {"n_channels": int(self.n_channels), "emission": self.emission.detach().cpu()}

    def _load_extra_payload(self, payload: dict, *, device: torch.device):
        """Load serialized emission parameters.

        Args:
            payload: Serialized payload dictionary.
            device: Target torch device.
        """
        self.n_channels = int(payload.get("n_channels", self.n_channels))
        with torch.no_grad():
            self.emission.data = payload["emission"].to(device=device, dtype=self.dtype)
        self._normalize_emission()

    def fit_em(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        *,
        device: torch.device,
        max_iter: int,
        tol: float,
        update_start: bool,
        update_trans: bool,
        update_emission: bool,
        verbose: bool,
        **kwargs,
    ) -> List[float]:
        """Run EM updates for a multi-channel Bernoulli HMM.

        Args:
            X: Observations array (N, L, C).
            coords: Coordinate array aligned to X.
            device: Torch device.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            update_start: Whether to update start probabilities.
            update_trans: Whether to update transitions.
            update_emission: Whether to update emission parameters.
            verbose: Whether to log progress.
            **kwargs: Additional implementation-specific kwargs.

        Returns:
            List of log-likelihood proxy values.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError("MultiBernoulliHMM expects X shape (N,L,C).")

        obs = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=self.dtype, device=device)
        mask = torch.tensor(~np.isnan(X), dtype=torch.bool, device=device)

        eps = float(self.eps)
        K = self.n_states
        N, L, C = obs.shape

        self._ensure_n_channels(C, device)

        hist: List[float] = []
        for it in range(1, int(max_iter) + 1):
            gamma = self._forward_backward(obs, mask)  # (N,L,K)

            ll_proxy = float(torch.sum(torch.log(torch.clamp(gamma.sum(dim=2), min=eps))).item())
            hist.append(ll_proxy)

            # expected start
            start_acc = gamma[:, 0, :].sum(dim=0)  # (K,)

            # transitions xi
            logB = self._log_emission(obs, mask)
            logA = torch.log(self.trans + eps)
            alpha = torch.empty((N, L, K), dtype=self.dtype, device=device)
            alpha[:, 0, :] = torch.log(self.start + eps).unsqueeze(0) + logB[:, 0, :]
            for t in range(1, L):
                prev = alpha[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)
                alpha[:, t, :] = _logsumexp(prev, dim=1) + logB[:, t, :]
            beta = torch.empty((N, L, K), dtype=self.dtype, device=device)
            beta[:, L - 1, :] = 0.0
            for t in range(L - 2, -1, -1):
                temp = logA.unsqueeze(0) + (logB[:, t + 1, :] + beta[:, t + 1, :]).unsqueeze(1)
                beta[:, t, :] = _logsumexp(temp, dim=2)

            trans_acc = torch.zeros((K, K), dtype=self.dtype, device=device)
            # valid timestep if at least one channel observed at both positions
            valid_pos = mask.any(dim=2)  # (N,L)
            for t in range(L - 1):
                valid_t = (valid_pos[:, t] & valid_pos[:, t + 1]).float().view(N, 1, 1)
                log_xi = (
                    alpha[:, t, :].unsqueeze(2)
                    + logA.unsqueeze(0)
                    + (logB[:, t + 1, :] + beta[:, t + 1, :]).unsqueeze(1)
                )
                log_norm = _logsumexp(log_xi.view(N, -1), dim=1).view(N, 1, 1)
                xi = (log_xi - log_norm).exp() * valid_t
                trans_acc += xi.sum(dim=0)

            # emission update per channel
            gamma_k = gamma.unsqueeze(-1)  # (N,L,K,1)
            obs_c = obs.unsqueeze(2)  # (N,L,1,C)
            mask_c = mask.unsqueeze(2).float()  # (N,L,1,C)

            emit_num = (gamma_k * obs_c * mask_c).sum(dim=(0, 1))  # (K,C)
            emit_den = (gamma_k * mask_c).sum(dim=(0, 1))  # (K,C)

            with torch.no_grad():
                if update_start:
                    new_start = start_acc + eps
                    self.start.data = new_start / new_start.sum()

                if update_trans:
                    new_trans = trans_acc + eps
                    rs = new_trans.sum(dim=1, keepdim=True)
                    rs[rs == 0.0] = 1.0
                    self.trans.data = new_trans / rs

                if update_emission:
                    new_em = (emit_num + eps) / (emit_den + 2.0 * eps)
                    self.emission.data = new_em.clamp(min=eps, max=1.0 - eps)

            self._normalize_params()
            self._normalize_emission()

            if verbose:
                logger.info(
                    "[MultiBernoulliHMM.fit] iter=%s ll_proxy=%.6f",
                    it,
                    hist[-1],
                )

            if len(hist) > 1 and abs(hist[-1] - hist[-2]) < float(tol):
                break

        return hist

    def adapt_emissions(
        self,
        X: np.ndarray,
        coords: Optional[np.ndarray] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        iters: Optional[int] = None,
        max_iter: Optional[int] = None,  # alias for your trainer
        verbose: bool = False,
        **kwargs,
    ):
        """Adapt emissions with legacy parameter names.

        Args:
            X: Observations array.
            coords: Optional coordinate array.
            device: Device specifier.
            iters: Number of iterations.
            max_iter: Alias for iters.
            verbose: Whether to log progress.
            **kwargs: Additional kwargs forwarded to BaseHMM.adapt_emissions.

        Returns:
            List of log-likelihood values.
        """
        if iters is None:
            iters = int(max_iter) if max_iter is not None else int(kwargs.pop("iters", 5))
        return super().adapt_emissions(
            np.asarray(X, dtype=float),
            coords if coords is not None else None,
            iters=int(iters),
            device=device,
            verbose=verbose,
        )

    def _ensure_n_channels(self, C: int, device: torch.device):
        """Expand emission parameters when channel count changes.

        Args:
            C: Target channel count.
            device: Torch device for the new parameters.
        """
        C = int(C)
        if C == self.n_channels:
            return
        with torch.no_grad():
            old = self.emission.detach().cpu().numpy()  # (K, Cold)
            K = old.shape[0]
            new = np.full((K, C), 0.5, dtype=float)
            m = min(old.shape[1], C)
            new[:, :m] = old[:, :m]
            if C > old.shape[1]:
                fill = old.mean(axis=1, keepdims=True)
                new[:, m:] = fill
            self.n_channels = C
            self.emission = nn.Parameter(
                torch.tensor(new, dtype=self.dtype, device=device), requires_grad=False
            )
            self._normalize_emission()


# =============================================================================
# Distance-binned transitions (single-channel only)
# =============================================================================


@register_hmm("single_distance_binned")
class DistanceBinnedSingleBernoulliHMM(SingleBernoulliHMM):
    """
    Transition matrix depends on binned distances between consecutive coords.

    Config keys:
      hmm_distance_bins: list[int] edges (bp)
      hmm_init_transitions_by_bin: optional (n_bins,K,K)
    """

    def __init__(
        self,
        n_states: int = 2,
        init_emission: Optional[Sequence[float]] = None,
        distance_bins: Optional[Sequence[int]] = None,
        init_trans_by_bin: Optional[Any] = None,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize a distance-binned transition HMM.

        Args:
            n_states: Number of hidden states.
            init_emission: Initial emission probabilities per state.
            distance_bins: Distance bin edges in base pairs.
            init_trans_by_bin: Initial transition matrices per bin.
            eps: Smoothing epsilon for probabilities.
            dtype: Torch dtype for parameters.
        """
        super().__init__(n_states=n_states, init_emission=init_emission, eps=eps, dtype=dtype)

        self.distance_bins = np.asarray(
            distance_bins if distance_bins is not None else [1, 5, 10, 25, 50, 100], dtype=int
        )
        self.n_bins = int(len(self.distance_bins) + 1)

        if init_trans_by_bin is None:
            base = self.trans.detach().cpu().numpy()
            tb = np.stack([base for _ in range(self.n_bins)], axis=0)
        else:
            tb = np.asarray(init_trans_by_bin, dtype=float)
            if tb.shape != (self.n_bins, self.n_states, self.n_states):
                base = self.trans.detach().cpu().numpy()
                tb = np.stack([base for _ in range(self.n_bins)], axis=0)

        self.trans_by_bin = nn.Parameter(torch.tensor(tb, dtype=self.dtype), requires_grad=False)
        self._normalize_trans_by_bin()

    @classmethod
    def from_config(cls, cfg, *, override=None, device=None):
        """Create a distance-binned HMM from config.

        Args:
            cfg: Configuration mapping or object.
            override: Override values to apply.
            device: Optional device specifier.

        Returns:
            Initialized DistanceBinnedSingleBernoulliHMM instance.
        """
        merged = cls._cfg_to_dict(cfg)
        if override:
            merged.update(override)

        n_states = int(merged.get("hmm_n_states", 2))
        eps = float(merged.get("hmm_eps", 1e-8))
        dtype = _resolve_dtype(merged.get("hmm_dtype", None))
        dtype = _coerce_dtype_for_device(dtype, device)  # <<< NEW
        init_em = merged.get("hmm_init_emission_probs", None)

        bins = merged.get("hmm_distance_bins", [1, 5, 10, 25, 50, 100])
        init_tb = merged.get("hmm_init_transitions_by_bin", None)

        model = cls(
            n_states=n_states,
            init_emission=init_em,
            distance_bins=bins,
            init_trans_by_bin=init_tb,
            eps=eps,
            dtype=dtype,
        )
        if device is not None:
            model.to(torch.device(device) if isinstance(device, str) else device)
        model._persisted_cfg = merged
        return model

    def _ensure_device_dtype(self, device=None) -> torch.device:
        """Move transition-by-bin parameters to the requested device/dtype."""
        device = super()._ensure_device_dtype(device)
        self.trans_by_bin.data = self.trans_by_bin.data.to(device=device, dtype=self.dtype)
        return device

    def _normalize_trans_by_bin(self):
        """Normalize transition matrices per distance bin in-place."""
        with torch.no_grad():
            tb = self.trans_by_bin.data.reshape(self.n_bins, self.n_states, self.n_states)
            tb = tb + self.eps
            rs = tb.sum(dim=2, keepdim=True)
            rs[rs == 0.0] = 1.0
            self.trans_by_bin.data = tb / rs

    def _extra_save_payload(self) -> dict:
        """Return extra payload data for serialization."""
        p = super()._extra_save_payload()
        p.update(
            {
                "distance_bins": torch.tensor(self.distance_bins, dtype=torch.long),
                "trans_by_bin": self.trans_by_bin.detach().cpu(),
            }
        )
        return p

    def _load_extra_payload(self, payload: dict, *, device: torch.device):
        """Load serialized distance-bin parameters.

        Args:
            payload: Serialized payload dictionary.
            device: Target torch device.
        """
        super()._load_extra_payload(payload, device=device)
        self.distance_bins = (
            payload.get("distance_bins", torch.tensor([1, 5, 10, 25, 50, 100]))
            .cpu()
            .numpy()
            .astype(int)
        )
        self.n_bins = int(len(self.distance_bins) + 1)
        with torch.no_grad():
            self.trans_by_bin.data = payload["trans_by_bin"].to(device=device, dtype=self.dtype)
        self._normalize_trans_by_bin()

    def _bin_index(self, coords: np.ndarray) -> np.ndarray:
        """Return per-step distance bin indices for coordinates.

        Args:
            coords: Coordinate array.

        Returns:
            Array of bin indices (length L-1).
        """
        d = np.diff(np.asarray(coords, dtype=int))
        return np.digitize(d, self.distance_bins, right=True)  # length L-1

    def _forward_backward(
        self, obs: torch.Tensor, mask: torch.Tensor, *, coords: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """Run forward-backward using distance-binned transitions.

        Args:
            obs: Observation tensor.
            mask: Observation mask.
            coords: Coordinate array.

        Returns:
            Posterior probabilities (gamma).
        """
        if coords is None:
            raise ValueError("Distance-binned HMM requires coords.")
        device = obs.device
        eps = float(self.eps)
        K = self.n_states

        coords = np.asarray(coords, dtype=int)
        bins = torch.tensor(self._bin_index(coords), dtype=torch.long, device=device)  # (L-1,)

        logB = self._log_emission(obs, mask)  # (N,L,K)
        logstart = torch.log(self.start + eps)
        logA_by_bin = torch.log(self.trans_by_bin + eps)  # (nb,K,K)

        N, L, _ = logB.shape
        alpha = torch.empty((N, L, K), dtype=self.dtype, device=device)
        alpha[:, 0, :] = logstart.unsqueeze(0) + logB[:, 0, :]

        for t in range(1, L):
            b = int(bins[t - 1].item()) if (t - 1) < bins.numel() else 0
            logA = logA_by_bin[b]
            prev = alpha[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)
            alpha[:, t, :] = _logsumexp(prev, dim=1) + logB[:, t, :]

        beta = torch.empty((N, L, K), dtype=self.dtype, device=device)
        beta[:, L - 1, :] = 0.0
        for t in range(L - 2, -1, -1):
            b = int(bins[t].item()) if t < bins.numel() else 0
            logA = logA_by_bin[b]
            temp = logA.unsqueeze(0) + (logB[:, t + 1, :] + beta[:, t + 1, :]).unsqueeze(1)
            beta[:, t, :] = _logsumexp(temp, dim=2)

        log_gamma = alpha + beta
        logZ = _logsumexp(log_gamma, dim=2).unsqueeze(2)
        return (log_gamma - logZ).exp()

    def _viterbi(
        self, obs: torch.Tensor, mask: torch.Tensor, *, coords: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """Run Viterbi decoding using distance-binned transitions.

        Args:
            obs: Observation tensor.
            mask: Observation mask.
            coords: Coordinate array.

        Returns:
            Decoded state sequence tensor.
        """
        if coords is None:
            raise ValueError("Distance-binned HMM requires coords.")
        device = obs.device
        eps = float(self.eps)
        K = self.n_states

        coords = np.asarray(coords, dtype=int)
        bins = torch.tensor(self._bin_index(coords), dtype=torch.long, device=device)  # (L-1,)

        logB = self._log_emission(obs, mask)
        logstart = torch.log(self.start + eps)
        logA_by_bin = torch.log(self.trans_by_bin + eps)

        N, L, _ = logB.shape
        delta = torch.empty((N, L, K), dtype=self.dtype, device=device)
        psi = torch.empty((N, L, K), dtype=torch.long, device=device)

        delta[:, 0, :] = logstart.unsqueeze(0) + logB[:, 0, :]
        psi[:, 0, :] = -1

        for t in range(1, L):
            b = int(bins[t - 1].item()) if (t - 1) < bins.numel() else 0
            logA = logA_by_bin[b]
            cand = delta[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)
            best_val, best_idx = cand.max(dim=1)
            delta[:, t, :] = best_val + logB[:, t, :]
            psi[:, t, :] = best_idx

        last_state = torch.argmax(delta[:, L - 1, :], dim=1)
        states = torch.empty((N, L), dtype=torch.long, device=device)
        states[:, L - 1] = last_state
        for t in range(L - 2, -1, -1):
            states[:, t] = psi[torch.arange(N, device=device), t + 1, states[:, t + 1]]
        return states

    def fit_em(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        *,
        device: torch.device,
        max_iter: int,
        tol: float,
        update_start: bool,
        update_trans: bool,
        update_emission: bool,
        verbose: bool,
        **kwargs,
    ) -> List[float]:
        """Run EM updates for distance-binned transitions.

        Args:
            X: Observations array (N, L).
            coords: Coordinate array aligned to X.
            device: Torch device.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            update_start: Whether to update start probabilities.
            update_trans: Whether to update transitions.
            update_emission: Whether to update emission parameters.
            verbose: Whether to log progress.
            **kwargs: Additional implementation-specific kwargs.

        Returns:
            List of log-likelihood proxy values.
        """
        # Keep this simple: use gamma for emissions; transitions-by-bin updated via xi (same pattern).
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("DistanceBinnedSingleBernoulliHMM expects X shape (N,L).")

        coords = np.asarray(coords, dtype=int)
        bins_np = self._bin_index(coords)  # (L-1,)

        obs = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=self.dtype, device=device)
        mask = torch.tensor(~np.isnan(X), dtype=torch.bool, device=device)

        eps = float(self.eps)
        K = self.n_states
        N, L = obs.shape

        hist: List[float] = []
        for it in range(1, int(max_iter) + 1):
            gamma = self._forward_backward(obs, mask, coords=coords)  # (N,L,K)
            ll_proxy = float(torch.sum(torch.log(torch.clamp(gamma.sum(dim=2), min=eps))).item())
            hist.append(ll_proxy)

            # expected start
            start_acc = gamma[:, 0, :].sum(dim=0)

            # compute alpha/beta for xi
            logB = self._log_emission(obs, mask)
            logstart = torch.log(self.start + eps)
            logA_by_bin = torch.log(self.trans_by_bin + eps)

            alpha = torch.empty((N, L, K), dtype=self.dtype, device=device)
            alpha[:, 0, :] = logstart.unsqueeze(0) + logB[:, 0, :]

            for t in range(1, L):
                b = int(bins_np[t - 1])
                logA = logA_by_bin[b]
                prev = alpha[:, t - 1, :].unsqueeze(2) + logA.unsqueeze(0)
                alpha[:, t, :] = _logsumexp(prev, dim=1) + logB[:, t, :]

            beta = torch.empty((N, L, K), dtype=self.dtype, device=device)
            beta[:, L - 1, :] = 0.0
            for t in range(L - 2, -1, -1):
                b = int(bins_np[t])
                logA = logA_by_bin[b]
                temp = logA.unsqueeze(0) + (logB[:, t + 1, :] + beta[:, t + 1, :]).unsqueeze(1)
                beta[:, t, :] = _logsumexp(temp, dim=2)

            trans_acc_by_bin = torch.zeros((self.n_bins, K, K), dtype=self.dtype, device=device)
            for t in range(L - 1):
                b = int(bins_np[t])
                logA = logA_by_bin[b]
                valid_t = (mask[:, t] & mask[:, t + 1]).float().view(N, 1, 1)
                log_xi = (
                    alpha[:, t, :].unsqueeze(2)
                    + logA.unsqueeze(0)
                    + (logB[:, t + 1, :] + beta[:, t + 1, :]).unsqueeze(1)
                )
                log_norm = _logsumexp(log_xi.view(N, -1), dim=1).view(N, 1, 1)
                xi = (log_xi - log_norm).exp() * valid_t
                trans_acc_by_bin[b] += xi.sum(dim=0)

            mask_f = mask.float().unsqueeze(-1)
            emit_num = (gamma * obs.unsqueeze(-1) * mask_f).sum(dim=(0, 1))
            emit_den = (gamma * mask_f).sum(dim=(0, 1))

            with torch.no_grad():
                if update_start:
                    new_start = start_acc + eps
                    self.start.data = new_start / new_start.sum()

                if update_trans:
                    tb = trans_acc_by_bin + eps
                    rs = tb.sum(dim=2, keepdim=True)
                    rs[rs == 0.0] = 1.0
                    self.trans_by_bin.data = tb / rs

                if update_emission:
                    new_em = (emit_num + eps) / (emit_den + 2.0 * eps)
                    self.emission.data = new_em.clamp(min=eps, max=1.0 - eps)

            self._normalize_params()
            self._normalize_emission()
            self._normalize_trans_by_bin()

            if verbose:
                logger.info(
                    "[DistanceBinnedSingle.fit] iter=%s ll_proxy=%.6f",
                    it,
                    hist[-1],
                )

            if len(hist) > 1 and abs(hist[-1] - hist[-2]) < float(tol):
                break

        return hist

    def adapt_emissions(
        self,
        X: np.ndarray,
        coords: Optional[np.ndarray] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        iters: Optional[int] = None,
        max_iter: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Adapt emissions with legacy parameter names.

        Args:
            X: Observations array.
            coords: Optional coordinate array.
            device: Device specifier.
            iters: Number of iterations.
            max_iter: Alias for iters.
            verbose: Whether to log progress.
            **kwargs: Additional kwargs forwarded to BaseHMM.adapt_emissions.

        Returns:
            List of log-likelihood values.
        """
        if iters is None:
            iters = int(max_iter) if max_iter is not None else int(kwargs.pop("iters", 5))
        return super().adapt_emissions(
            np.asarray(X, dtype=float),
            coords if coords is not None else None,
            iters=int(iters),
            device=device,
            verbose=verbose,
        )


# =============================================================================
# Facade class to match workflow import style
# =============================================================================


class HMM:
    """
    Facade so workflow can do:
      from ..hmm.HMM import HMM
      hmm = HMM.from_config(cfg, arch="single")
      hmm.save(...)
      hmm = HMM.load(...)
    """

    @staticmethod
    def from_config(cfg, arch: Optional[str] = None, **kwargs) -> BaseHMM:
        """Create an HMM instance from configuration.

        Args:
            cfg: Configuration mapping or object.
            arch: Optional HMM architecture name.
            **kwargs: Additional parameters passed to the factory.

        Returns:
            Initialized HMM instance.
        """
        return create_hmm(cfg, arch=arch, **kwargs)

    @staticmethod
    def load(path: Union[str, Path], device: Optional[Union[str, torch.device]] = None) -> BaseHMM:
        """Load an HMM instance from disk.

        Args:
            path: Path to the serialized model.
            device: Optional device specifier.

        Returns:
            Loaded HMM instance.
        """
        return BaseHMM.load(path, device=device)
