"""Per-read periodicity over a region (`RPG-02`): pure statistics.

One Lomb-Scargle periodogram per (read, region), on the direct signal --
binary calls or a layer's values at observed positions -- with the spatial
stage's method (`analyze_ls_periodicity_direct`): polynomial detrend, power
on a 1 bp period grid, the peak within a peak-search range, SNR, FWHM.

The period range narrows to the region: a periodogram needs several cycles of
the longest period it reports, so the longest period kept is
``min(requested_max, region_length / min_cycles)`` and the peak range is
clipped to it. A region too short for the requested minimum period or the
peak range is skipped (status ``region_too_short``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smftools.analysis.compute.ls_periodicity import (
    LS_POLY_DEGREE,
    MIN_SITES_PER_READ,
    analyze_ls_periodicity_direct,
)

PERIOD_RANGE_BP = (80.0, 400.0)
PEAK_RANGE_BP = (150.0, 250.0)
MIN_CYCLES = 3.0
MIN_COVERAGE = 0.8
STATISTICS = ("peak_period_bp", "snr", "peak_power", "peak_power_raw", "fwhm_bp")
# Per-read status values.
OK = "ok"
REGION_TOO_SHORT = "region_too_short"
LOW_COVERAGE = "low_coverage"
TOO_FEW_SITES = "too_few_sites"
NO_SIGNAL = "no_signal"  # flat after detrending, or no peak in range


@dataclass(frozen=True)
class PeriodGrid:
    """One region's period axis (descending, as the spatial stage stores it)."""

    start: int
    end: int
    periods: np.ndarray
    period_range: tuple[float, float]
    peak_range: tuple[float, float]
    requested_period_range: tuple[float, float]
    requested_peak_range: tuple[float, float]
    status: str  # OK or REGION_TOO_SHORT

    @property
    def name(self) -> str:
        return f"{self.start}-{self.end}"

    @property
    def narrowed(self) -> bool:
        return self.period_range != self.requested_period_range

    def record(self) -> dict:
        return {
            "region": self.name,
            "start": self.start,
            "end": self.end,
            "length_bp": self.end - self.start,
            "status": self.status,
            "period_min_bp": self.period_range[0],
            "period_max_bp": self.period_range[1],
            "peak_min_bp": self.peak_range[0],
            "peak_max_bp": self.peak_range[1],
            "requested_period_max_bp": self.requested_period_range[1],
            "requested_peak_max_bp": self.requested_peak_range[1],
            "narrowed": self.narrowed,
            "n_periods": int(self.periods.size),
        }


def period_grid(
    start: int,
    end: int,
    *,
    period_range: tuple[float, float] = PERIOD_RANGE_BP,
    peak_range: tuple[float, float] = PEAK_RANGE_BP,
    min_cycles: float = MIN_CYCLES,
) -> PeriodGrid:
    """The region's period grid, narrowed so ``min_cycles`` of the longest period fit."""
    if end <= start:
        raise ValueError(f"region {start}-{end} is empty")
    if min_cycles <= 0:
        raise ValueError("min_cycles must be positive")
    low, high = (float(value) for value in period_range)
    peak_low, peak_high = (float(value) for value in peak_range)
    if not 0 < low < high:
        raise ValueError(f"period range {period_range} must be increasing and positive")
    if not low <= peak_low < peak_high <= high:
        raise ValueError(f"peak range {peak_range} must lie inside period range {period_range}")
    longest = float(np.floor(min(high, (end - start) / min_cycles)))
    clipped_peak_high = min(peak_high, longest)
    if longest < low or clipped_peak_high <= peak_low:
        status, periods = REGION_TOO_SHORT, np.array([], dtype=float)
        effective, effective_peak = (low, max(low, longest)), (peak_low, clipped_peak_high)
    else:
        status = OK
        periods = np.arange(longest, low - 1, -1, dtype=float)
        effective, effective_peak = (low, longest), (peak_low, clipped_peak_high)
    return PeriodGrid(
        start=int(start),
        end=int(end),
        periods=periods,
        period_range=effective,
        peak_range=effective_peak,
        requested_period_range=(low, high),
        requested_peak_range=(peak_low, peak_high),
        status=status,
    )


def read_periodograms(
    positions: np.ndarray,
    values: np.ndarray,
    observed: np.ndarray,
    design: np.ndarray,
    grid: PeriodGrid,
    *,
    poly_degree: int = LS_POLY_DEGREE,
    min_sites: int = MIN_SITES_PER_READ,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Periodograms of every read over ``grid``'s region.

    ``positions`` are the columns' coordinates; ``values``, ``observed`` and
    ``design`` are reads x columns. Only columns inside the region are used.
    A read needs ``min_coverage`` of the region's design positions observed and
    ``min_sites`` observed values. Returns power (reads x periods, NaN where
    not scored) and per-read statistics with a ``status``.
    """
    positions = np.asarray(positions)
    inside = (positions >= grid.start) & (positions < grid.end)
    x = positions[inside].astype(float)
    values = np.asarray(values, dtype=float)[:, inside]
    observed = np.asarray(observed, dtype=bool)[:, inside]
    design = np.asarray(design, dtype=bool)[:, inside]
    n_reads = values.shape[0]
    power = np.full((n_reads, grid.periods.size), np.nan, dtype=np.float32)
    stats = {name: np.full(n_reads, np.nan) for name in STATISTICS}
    n_sites = observed.sum(axis=1).astype(np.int64)
    designed = design.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.where(designed > 0, n_sites / designed, 0.0)
    status = np.full(n_reads, OK, dtype=object)
    if grid.status != OK:
        status[:] = grid.status
    else:
        status[coverage < min_coverage] = LOW_COVERAGE
        status[(status == OK) & (n_sites < min_sites)] = TOO_FEW_SITES
        for row in np.flatnonzero(status == OK):
            signal = np.where(observed[row], values[row], np.nan)
            result = analyze_ls_periodicity_direct(
                x,
                signal,
                nrl_search_bp=grid.peak_range,
                period_range_bp=grid.period_range,
                poly_degree=poly_degree,
                min_sites=min_sites,
            )
            if result is None:
                status[row] = NO_SIGNAL
                continue
            power[row] = np.asarray(result["ls_power"], dtype=np.float32)
            stats["peak_period_bp"][row] = float(result["ls_nrl_bp"])
            stats["snr"][row] = float(result["ls_snr"])
            stats["peak_power"][row] = float(result["ls_peak_power"])
            stats["peak_power_raw"][row] = float(result["ls_peak_power_raw"])
            stats["fwhm_bp"][row] = float(result["ls_fwhm_bp"])
    table = pd.DataFrame(
        {
            "region": grid.name,
            "status": status.astype(str),
            "n_sites": n_sites,
            "coverage": coverage,
            **stats,
        }
    )
    return power, table
