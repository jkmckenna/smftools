"""Small, interpretable convolutional scanners (`MLR-09`, spatial class).

A stack of convolution layers (one or a few, few filters), optionally
downsampled between layers by max pooling, then a pooling of each final
filter over the molecule and a linear (or small hidden-layer) head:

- ``pooling`` ``("max",)`` -- "does the pattern occur anywhere": with one
  layer and a linear head, each filter's weights are its pattern and its
  contribution to a molecule's logit is its peak activation times its head
  weight (the classic motif scanner);
- ``"avg"`` -- how much of the pattern over the read; ``"attention"`` -- a
  content-weighted average; several may be combined;
- ``adaptive_bins`` > 0 instead pools each filter into that many equal bins
  along the molecule, bringing back coarse position (where a pattern matters).

Inputs follow the residual CNN's contract (channel-first values plus validity
masks, `MaskedConvInputs`: mask channels, span masking). ``receptive_field``
is the widest input window one final-layer position sees; ``feature_stride``
is how many input positions one final-layer position steps (the product of
the downsampling factors), so detector catalogues and effective spans map
feature positions back to input positions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from smftools.optional_imports import require

from .residual_cnn import AttentionPooling1d, MaskedConvInputs, pooling_mask

torch = require("torch", extra="ml-base", purpose="convolutional scanner models")
nn = torch.nn
F = torch.nn.functional

POOLINGS = ("max", "avg", "attention")


class ConvScannerConfigError(ValueError):
    """Raised when a convolutional scanner architecture is invalid."""


def _positive(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConvScannerConfigError(f"{path} must be a positive integer")
    return value


def _positives(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ConvScannerConfigError(f"{path} must be a non-empty sequence")
    return tuple(_positive(item, f"{path}[]") for item in value)


@dataclass(frozen=True)
class ConvScannerConfig:
    """Exact architecture parameters of a convolutional scanner."""

    in_channels: int
    filters: tuple[int, ...] = (16,)
    kernel_sizes: tuple[int, ...] = (21,)
    dilations: tuple[int, ...] | None = None  # default 1 per layer
    downsample: int = 1  # max-pool factor between layers (1: none)
    pooling: tuple[str, ...] = ("max",)
    adaptive_bins: int = 0  # > 0: adaptive max pooling into bins instead of `pooling`
    head_hidden: int = 0  # 0: linear head
    dropout: float = 0.0
    batch_norm: bool = True
    output_dim: int = 1
    mask_channels: bool = True
    span_masking: bool = True
    max_receptive_field: int | None = None

    def __post_init__(self) -> None:
        _positive(self.in_channels, "in_channels")
        filters = _positives(self.filters, "filters")
        kernels = _positives(self.kernel_sizes, "kernel_sizes")
        dilations = (
            (1,) * len(filters)
            if self.dilations is None
            else _positives(self.dilations, "dilations")
        )
        if not len(filters) == len(kernels) == len(dilations):
            raise ConvScannerConfigError(
                "filters, kernel_sizes and dilations need one value per layer"
            )
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ConvScannerConfigError("kernel_sizes must be odd to keep positions centred")
        object.__setattr__(self, "filters", filters)
        object.__setattr__(self, "kernel_sizes", kernels)
        object.__setattr__(self, "dilations", dilations)
        _positive(self.downsample, "downsample")
        if self.adaptive_bins < 0 or isinstance(self.adaptive_bins, bool):
            raise ConvScannerConfigError("adaptive_bins must be a non-negative integer")
        pooling = tuple(self.pooling)
        if not pooling or set(pooling) - set(POOLINGS) or len(set(pooling)) != len(pooling):
            raise ConvScannerConfigError(f"pooling must be distinct values of {list(POOLINGS)}")
        object.__setattr__(self, "pooling", pooling)
        if self.head_hidden < 0 or isinstance(self.head_hidden, bool):
            raise ConvScannerConfigError("head_hidden must be a non-negative integer")
        if not 0 <= float(self.dropout) < 1:
            raise ConvScannerConfigError("dropout must be in [0, 1)")
        object.__setattr__(self, "dropout", float(self.dropout))
        for name in ("batch_norm", "mask_channels", "span_masking"):
            if not isinstance(getattr(self, name), bool):
                raise ConvScannerConfigError(f"{name} must be boolean")
        _positive(self.output_dim, "output_dim")
        if self.max_receptive_field is not None:
            limit = _positive(self.max_receptive_field, "max_receptive_field")
            if self.receptive_field > limit:
                raise ConvScannerConfigError(
                    f"receptive field {self.receptive_field} exceeds max_receptive_field {limit}"
                )

    @property
    def receptive_field(self) -> int:
        """Widest input window (positions) one final-layer position sees."""
        field, jump = 1, 1
        for index, (kernel, dilation) in enumerate(zip(self.kernel_sizes, self.dilations)):
            field += (kernel - 1) * dilation * jump
            if index < len(self.kernel_sizes) - 1 and self.downsample > 1:
                field += (self.downsample - 1) * jump
                jump *= self.downsample
        return field

    @property
    def feature_stride(self) -> int:
        """Input positions per final-layer position."""
        return self.downsample ** (len(self.filters) - 1)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("filters", "kernel_sizes", "dilations", "pooling"):
            payload[name] = list(payload[name])
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ConvScannerConfig:
        allowed = {field for field in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - allowed)
        if unknown or "in_channels" not in raw:
            raise ConvScannerConfigError(
                f"conv scanner fields must be among {sorted(allowed)} (unknown {unknown})"
            )
        return cls(**dict(raw))


class ConvScanner1d(MaskedConvInputs, nn.Module):
    """Conv layers (optional downsampling) -> per-filter pooling -> head."""

    def __init__(self, config: ConvScannerConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_stride = config.feature_stride
        layers = []
        channels = config.in_channels * (2 if config.mask_channels else 1)
        for out_channels, kernel, dilation in zip(
            config.filters, config.kernel_sizes, config.dilations
        ):
            layers.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            channels,
                            out_channels,
                            kernel_size=kernel,
                            dilation=dilation,
                            padding=(kernel // 2) * dilation,
                        ),
                        "norm": nn.BatchNorm1d(out_channels)
                        if config.batch_norm
                        else nn.Identity(),
                    }
                )
            )
            channels = out_channels
        self.layers = nn.ModuleList(layers)
        self.attn_pool = (
            AttentionPooling1d(channels)
            if "attention" in config.pooling and not config.adaptive_bins
            else None
        )
        summary = channels * (config.adaptive_bins if config.adaptive_bins else len(config.pooling))
        if config.head_hidden:
            self.head = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(summary, config.head_hidden),
                nn.ReLU(),
                nn.Linear(config.head_hidden, config.output_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(config.dropout), nn.Linear(summary, config.output_dim)
            )

    @property
    def attribution_layer(self):
        """The final convolution, for layer attribution."""
        return self.layers[-1]["conv"]

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
        valid = position_valid
        for index, layer in enumerate(self.layers):
            values = F.relu(layer["norm"](layer["conv"](values)))
            values = values.masked_fill(~valid[:, None, :], 0.0)
            if index < len(self.layers) - 1 and self.config.downsample > 1:
                values = F.max_pool1d(values, self.config.downsample, ceil_mode=True)
                valid = (
                    F.max_pool1d(valid[:, None, :].float(), self.config.downsample, ceil_mode=True)[
                        :, 0
                    ]
                    > 0
                )
        return values, valid

    def forward_features(
        self,
        values,
        *,
        observed_mask=None,
        availability_mask=None,
        design_mask=None,
        padding_mask=None,
    ):
        """Final-layer filter activations (batch, filters, positions / stride)."""
        features, _valid = self._forward_features_and_mask(
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
        """``(batch, output_dim)`` logits."""
        features, valid = self._forward_features_and_mask(
            values,
            observed_mask=observed_mask,
            availability_mask=availability_mask,
            design_mask=design_mask,
            padding_mask=padding_mask,
        )
        if self.config.adaptive_bins:
            pooled = F.adaptive_max_pool1d(features, self.config.adaptive_bins).flatten(1)
            return self.head(pooled)
        mask = pooling_mask(valid)[:, None, :]
        parts = []
        for kind in self.config.pooling:
            if kind == "max":
                parts.append(features.masked_fill(~mask, float("-inf")).max(dim=-1).values)
            elif kind == "avg":
                parts.append(features.sum(dim=-1) / valid[:, None, :].sum(dim=-1).clamp(min=1))
            else:
                parts.append(self.attn_pool(features, position_mask=valid))
        return self.head(torch.cat(parts, dim=1))


def build_conv_scanner(config: ConvScannerConfig):
    """Construct a convolutional scanner from a validated configuration."""
    return ConvScanner1d(config)
