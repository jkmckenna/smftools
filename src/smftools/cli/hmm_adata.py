from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from smftools.logging_utils import get_logger

# FIX: import _to_dense_np to avoid NameError
from ..hmm.HMM import _safe_int_coords, _to_dense_np, create_hmm, normalize_hmm_feature_sets

logger = get_logger(__name__)

# =============================================================================
# Helpers: extracting training arrays
# =============================================================================


def _get_training_matrix(
    subset, cols_mask: np.ndarray, smf_modality: Optional[str], cfg
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Matches your existing behavior:
      - direct -> uses cfg.output_binary_layer_name in .layers
      - else   -> uses .X
    Returns (X, layer_name_or_None) where X is dense float array.
    """
    sub = subset[:, cols_mask]
    if smf_modality == "direct":
        hmm_layer = getattr(cfg, "output_binary_layer_name", None)
        if hmm_layer is None or hmm_layer not in sub.layers:
            raise KeyError(f"Missing HMM training layer '{hmm_layer}' in subset.")
        mat = sub.layers[hmm_layer]
    else:
        hmm_layer = None
        mat = sub.X

    X = _to_dense_np(mat).astype(float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D training matrix; got {X.shape}")
    return X, hmm_layer


def _resolve_pos_mask_for_methbase(subset, ref: str, methbase: str) -> Optional[np.ndarray]:
    """
    Reproduces your mask resolution, with compatibility for both *_any_C_site and *_C_site.
    """
    key = str(methbase).strip().lower()

    if key in ("a",):
        col = f"{ref}_A_site"
        if col not in subset.var:
            return None
        return np.asarray(subset.var[col] == "A")

    if key in ("c", "any_c", "anyc", "any-c"):
        for col in (f"{ref}_any_C_site", f"{ref}_C_site"):
            if col in subset.var:
                return np.asarray(subset.var[col])
        return None

    if key in ("gpc", "gpc_site", "gpc-site"):
        col = f"{ref}_GpC_site"
        if col not in subset.var:
            return None
        return np.asarray(subset.var[col])

    if key in ("cpg", "cpg_site", "cpg-site"):
        col = f"{ref}_CpG_site"
        if col not in subset.var:
            return None
        return np.asarray(subset.var[col])

    alt = f"{ref}_{methbase}_site"
    if alt not in subset.var:
        return None
    return np.asarray(subset.var[alt])


def build_single_channel(
    subset, ref: str, methbase: str, smf_modality: Optional[str], cfg
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X     (N, Lmb) float with NaNs allowed
      coords (Lmb,) int coords from var_names
    """
    pm = _resolve_pos_mask_for_methbase(subset, ref, methbase)
    if pm is None or int(np.sum(pm)) == 0:
        raise ValueError(f"No columns for methbase={methbase} on ref={ref}")
    X, _ = _get_training_matrix(subset, pm, smf_modality, cfg)
    coords, _ = _safe_int_coords(subset[:, pm].var_names)
    return X, coords


def build_multi_channel_union(
    subset, ref: str, methbases: Sequence[str], smf_modality: Optional[str], cfg
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (N, Lunion, C) on union coordinate grid across methbases.

    Returns:
      X3d:    (N, Lunion, C) float with NaN where methbase has no site
      coords: (Lunion,) int union coords
      used_methbases: list of methbases actually included (>=2)
    """
    per: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []  # (mb, X, coords, pm)

    for mb in methbases:
        pm = _resolve_pos_mask_for_methbase(subset, ref, mb)
        if pm is None or int(np.sum(pm)) == 0:
            continue
        Xmb, _ = _get_training_matrix(subset, pm, smf_modality, cfg)  # (N,Lmb)
        cmb, _ = _safe_int_coords(subset[:, pm].var_names)
        per.append((mb, Xmb.astype(float), cmb.astype(int), pm))

    if len(per) < 2:
        raise ValueError(f"Need >=2 methbases with columns for union multi-channel on ref={ref}")

    # union coordinates
    coords = np.unique(np.concatenate([c for _, _, c, _ in per], axis=0)).astype(int)
    idx = {int(v): i for i, v in enumerate(coords.tolist())}

    N = per[0][1].shape[0]
    L = coords.shape[0]
    C = len(per)
    X3 = np.full((N, L, C), np.nan, dtype=float)

    for ci, (mb, Xmb, cmb, _) in enumerate(per):
        cols = np.array([idx[int(v)] for v in cmb.tolist()], dtype=int)
        X3[:, cols, ci] = Xmb

    used = [mb for (mb, _, _, _) in per]
    return X3, coords, used


@dataclass
class HMMTask:
    name: str
    signals: List[str]  # e.g. ["GpC"] or ["GpC","CpG"] or ["CpG"]
    feature_groups: List[str]  # e.g. ["footprint","accessible"] or ["cpg"]
    output_prefix: Optional[str] = None  # force prefix (CpG task uses "CpG")


def build_hmm_tasks(cfg: Union[dict, Any]) -> List[HMMTask]:
    """
    Accessibility signals come from cfg['hmm_methbases'].
    CpG task is enabled by cfg['cpg']==True, independent of hmm_methbases.
    """
    if not isinstance(cfg, dict):
        # best effort conversion
        cfg = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}

    tasks: List[HMMTask] = []

    # accessibility task
    methbases = list(cfg.get("hmm_methbases", []) or [])
    if len(methbases) > 0:
        tasks.append(
            HMMTask(
                name="accessibility",
                signals=methbases,
                feature_groups=["footprint", "accessible"],
                output_prefix=None,
            )
        )

    # CpG task (special case)
    if bool(cfg.get("cpg", False)):
        tasks.append(
            HMMTask(
                name="cpg",
                signals=["CpG"],
                feature_groups=["cpg"],
                output_prefix="CpG",
            )
        )

    return tasks


def select_hmm_arch(cfg: dict, signals: Sequence[str]) -> str:
    """
    Simple, explicit model selection:
      - distance-aware => 'single_distance_binned' (only meaningful for single-channel)
      - multi-signal   => 'multi'
      - else           => 'single'
    """
    if bool(cfg.get("hmm_distance_aware", False)) and len(signals) == 1:
        return "single_distance_binned"
    if len(signals) > 1:
        return "multi"
    return "single"


def resolve_input_layer(adata, cfg: dict, layer_override: Optional[str]) -> Optional[str]:
    """
    If direct modality, prefer cfg.output_binary_layer_name.
    Else use layer_override or None (meaning use .X).
    """
    smf_modality = cfg.get("smf_modality", None)
    if smf_modality == "direct":
        nm = cfg.get("output_binary_layer_name", None)
        if nm is None:
            raise KeyError("cfg.output_binary_layer_name missing for smf_modality='direct'")
        if nm not in adata.layers:
            raise KeyError(f"Direct modality expects layer '{nm}' in adata.layers")
        return nm
    return layer_override


def _ensure_layer_and_assign_rows(adata, layer_name: str, row_mask: np.ndarray, subset_layer):
    """
    Writes subset_layer (n_subset_obs, n_vars) into adata.layers[layer_name] for rows where row_mask==True.
    """
    row_mask = np.asarray(row_mask, dtype=bool)
    if row_mask.ndim != 1 or row_mask.size != adata.n_obs:
        raise ValueError("row_mask must be length adata.n_obs")

    arr = _to_dense_np(subset_layer)
    if arr.shape != (int(row_mask.sum()), adata.n_vars):
        raise ValueError(
            f"subset layer '{layer_name}' shape {arr.shape} != ({int(row_mask.sum())}, {adata.n_vars})"
        )

    if layer_name not in adata.layers:
        adata.layers[layer_name] = np.zeros((adata.n_obs, adata.n_vars), dtype=arr.dtype)

    adata.layers[layer_name][row_mask, :] = arr


def resolve_torch_device(device_str: str | None) -> torch.device:
    d = (device_str or "auto").lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


# =============================================================================
# Model selection + fit strategy manager
# =============================================================================
@dataclass
class HMMTrainer:
    cfg: Any
    models_dir: Path

    def __post_init__(self):
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def choose_arch(self, *, multichannel: bool) -> str:
        use_dist = bool(getattr(self.cfg, "hmm_distance_aware", False))
        if multichannel:
            return "multi"
        return "single_distance_binned" if use_dist else "single"

    def _fit_scope(self) -> str:
        return str(getattr(self.cfg, "hmm_fit_scope", "per_sample")).lower()
        # "per_sample" | "global" | "global_then_adapt"

    def _path(self, kind: str, sample: str, ref: str, label: str) -> Path:
        # kind: "GLOBAL" | "PER" | "ADAPT"
        def safe(s):
            str(s).replace("/", "_")

        return self.models_dir / f"{kind}_{safe(sample)}_{safe(ref)}_{safe(label)}.pt"

    def _save(self, model, path: Path):
        override = {}
        if getattr(model, "hmm_name", None) == "multi":
            override["hmm_n_channels"] = int(getattr(model, "n_channels", 2))
        if getattr(model, "hmm_name", None) == "single_distance_binned":
            override["hmm_distance_bins"] = list(
                getattr(model, "distance_bins", [1, 5, 10, 25, 50, 100])
            )

        payload = {
            "state_dict": model.state_dict(),
            "hmm_arch": getattr(model, "hmm_name", None) or getattr(self.cfg, "hmm_arch", None),
            "override": override,
        }
        torch.save(payload, path)

    def _load(self, path: Path, arch: str, device):
        payload = torch.load(path, map_location="cpu")
        override = payload.get("override", None)
        m = create_hmm(self.cfg, arch=arch, override=override, device=device)
        sd = payload["state_dict"]

        target_dtype = next(m.parameters()).dtype
        for k, v in list(sd.items()):
            if isinstance(v, torch.Tensor) and v.dtype != target_dtype:
                sd[k] = v.to(dtype=target_dtype)

        m.load_state_dict(sd)
        m.to(device)
        m.eval()
        return m

    def fit_or_load(
        self,
        *,
        sample: str,
        ref: str,
        label: str,
        arch: str,
        X,
        coords: Optional[np.ndarray],
        device,
    ):
        force_fit = bool(getattr(self.cfg, "force_redo_hmm_fit", False))
        scope = self._fit_scope()

        max_iter = int(getattr(self.cfg, "hmm_max_iter", 50))
        tol = float(getattr(self.cfg, "hmm_tol", 1e-4))
        verbose = bool(getattr(self.cfg, "hmm_verbose", False))

        # ---- global then adapt ----
        if scope == "global_then_adapt":
            p_global = self._path("GLOBAL", "ALL", ref, label)
            if p_global.exists() and not force_fit:
                base = self._load(p_global, arch=arch, device=device)
            else:
                base = create_hmm(self.cfg, arch=arch).to(device)
                if arch == "single_distance_binned":
                    base.fit(
                        X, device=device, coords=coords, max_iter=max_iter, tol=tol, verbose=verbose
                    )
                else:
                    base.fit(X, device=device, max_iter=max_iter, tol=tol, verbose=verbose)
                self._save(base, p_global)

            p_adapt = self._path("ADAPT", sample, ref, label)
            if p_adapt.exists() and not force_fit:
                return self._load(p_adapt, arch=arch, device=device)

            # IMPORTANT: this assumes you added model.adapt_emissions(...)
            adapted = copy.deepcopy(base).to(device)
            if arch == "single_distance_binned":
                adapted.adapt_emissions(
                    X,
                    coords,
                    device=device,
                    max_iter=int(getattr(self.cfg, "hmm_adapt_iters", 10)),
                    verbose=verbose,
                )

            else:
                adapted.adapt_emissions(
                    X,
                    coords,
                    device=device,
                    max_iter=int(getattr(self.cfg, "hmm_adapt_iters", 10)),
                    verbose=verbose,
                )

            self._save(adapted, p_adapt)
            return adapted

        # ---- global only ----
        if scope == "global":
            p = self._path("GLOBAL", "ALL", ref, label)
            if p.exists() and not force_fit:
                return self._load(p, arch=arch, device=device)

        # ---- per sample ----
        else:
            p = self._path("PER", sample, ref, label)
            if p.exists() and not force_fit:
                return self._load(p, arch=arch, device=device)

        m = create_hmm(self.cfg, arch=arch, device=device)
        if arch == "single_distance_binned":
            m.fit(X, coords, device=device, max_iter=max_iter, tol=tol, verbose=verbose)
        else:
            m.fit(X, coords, device=device, max_iter=max_iter, tol=tol, verbose=verbose)
        self._save(m, p)
        return m


def _fully_qualified_merge_layers(cfg, prefix: str) -> List[Tuple[str, int]]:
    """
    cfg.hmm_merge_layer_features is assumed to be a list of (core_layer_name, merge_distance),
    where core_layer_name is like "all_accessible_features" (NOT prefixed with methbase).
    We expand to f"{prefix}_{core_layer_name}".
    """
    out = []
    for core_layer, dist in getattr(cfg, "hmm_merge_layer_features", []) or []:
        if not core_layer:
            continue
        out.append((f"{prefix}_{core_layer}", int(dist)))
    return out


def hmm_adata(config_path: str):
    """
    CLI-facing wrapper for HMM analysis.

    Command line entrypoint:
        smftools hmm <config_path>

    Responsibilities:
    - Build cfg via load_adata()
    - Ensure preprocess + spatial stages are run
    - Decide which AnnData to start from (hmm > spatial > pp_dedup > pp > raw)
    - Call hmm_adata_core(cfg, adata, paths)
    """
    from ..readwrite import safe_read_h5ad
    from .helpers import get_adata_paths
    from .load_adata import load_adata
    from .preprocess_adata import preprocess_adata
    from .spatial_adata import spatial_adata

    # 1) load cfg / stage paths
    _, _, cfg = load_adata(config_path)
    paths = get_adata_paths(cfg)

    # 2) make sure upstream stages are run (they have their own skipping logic)
    preprocess_adata(config_path)
    spatial_ad, _ = spatial_adata(config_path)

    # 3) choose starting AnnData
    # Prefer:
    #   - existing HMM h5ad if not forcing redo
    #   - in-memory spatial_ad from wrapper call
    #   - saved spatial / pp_dedup / pp / raw on disk
    if paths.hmm.exists() and not (cfg.force_redo_hmm_fit or cfg.force_redo_hmm_apply):
        adata, _ = safe_read_h5ad(paths.hmm)
        return adata, paths.hmm

    if spatial_ad is not None:
        adata = spatial_ad
    elif paths.spatial.exists():
        adata, _ = safe_read_h5ad(paths.spatial)
    elif paths.pp_dedup.exists():
        adata, _ = safe_read_h5ad(paths.pp_dedup)
    elif paths.pp.exists():
        adata, _ = safe_read_h5ad(paths.pp)
    elif paths.raw.exists():
        adata, _ = safe_read_h5ad(paths.raw)
    else:
        raise FileNotFoundError(
            "No AnnData available for HMM: expected at least raw or preprocessed h5ad."
        )

    # 4) delegate to core
    adata, hmm_adata_path = hmm_adata_core(cfg, adata, paths)
    return adata, hmm_adata_path


def hmm_adata_core(cfg, adata, paths) -> Tuple["anndata.AnnData", Path]:
    """
    Core HMM analysis pipeline.

    Assumes:
    - cfg is an ExperimentConfig
    - adata is the starting AnnData (typically spatial + dedup)
    - paths is an AdataPaths object (with .raw/.pp/.pp_dedup/.spatial/.hmm)

    Does NOT decide which h5ad to start from – that is the wrapper's job.
    """

    import numpy as np

    from ..hmm import call_hmm_peaks
    from ..plotting import (
        combined_hmm_raw_clustermap,
        plot_hmm_layers_rolling_by_sample_ref,
        plot_hmm_size_contours,
    )
    from ..readwrite import make_dirs
    from .helpers import write_gz_h5ad

    smf_modality = cfg.smf_modality
    deaminase = smf_modality == "deaminase"

    output_directory = Path(cfg.output_directory)
    make_dirs([output_directory])

    pp_dir = output_directory / "preprocessed" / "deduplicated"

    # ---------------------------- HMM annotate stage ----------------------------
    if not (cfg.bypass_hmm_fit and cfg.bypass_hmm_apply):
        hmm_models_dir = pp_dir / "10_hmm_models"
        make_dirs([pp_dir, hmm_models_dir])

        # Standard bookkeeping
        uns_key = "hmm_appended_layers"
        if adata.uns.get(uns_key) is None:
            adata.uns[uns_key] = []
        global_appended = list(adata.uns.get(uns_key, []))

        # Prepare trainer + feature config
        trainer = HMMTrainer(cfg=cfg, models_dir=hmm_models_dir)

        feature_sets = normalize_hmm_feature_sets(getattr(cfg, "hmm_feature_sets", None))
        prob_thr = float(getattr(cfg, "hmm_feature_prob_threshold", 0.5))
        decode = str(getattr(cfg, "hmm_decode", "marginal"))
        write_post = bool(getattr(cfg, "hmm_write_posterior", True))
        post_state = getattr(cfg, "hmm_posterior_state", "Modified")
        merged_suffix = str(getattr(cfg, "hmm_merged_suffix", "_merged"))
        force_apply = bool(getattr(cfg, "force_redo_hmm_apply", False))
        bypass_apply = bool(getattr(cfg, "bypass_hmm_apply", False))
        bypass_fit = bool(getattr(cfg, "bypass_hmm_fit", False))

        samples = adata.obs[cfg.sample_name_col_for_plotting].cat.categories
        references = adata.obs[cfg.reference_column].cat.categories
        methbases = list(getattr(cfg, "hmm_methbases", [])) or []

        if not methbases:
            raise ValueError("cfg.hmm_methbases is empty.")

        # Top-level skip
        already = bool(adata.uns.get("hmm_annotated", False))
        if already and not (bool(getattr(cfg, "force_redo_hmm_fit", False)) or force_apply):
            pass

        else:
            for sample in samples:
                for ref in references:
                    mask = (adata.obs[cfg.sample_name_col_for_plotting] == sample) & (
                        adata.obs[cfg.reference_column] == ref
                    )
                    if int(np.sum(mask)) == 0:
                        continue

                    subset = adata[mask].copy()
                    subset.uns[uns_key] = []  # isolate appended tracking per subset

                    # ---- Decide which tasks to run ----
                    methbases = list(getattr(cfg, "hmm_methbases", [])) or []
                    run_multi = bool(getattr(cfg, "hmm_run_multichannel", True))
                    run_cpg = bool(getattr(cfg, "cpg", False))
                    device = resolve_torch_device(cfg.device)

                    # ---- split feature sets ----
                    feature_sets_all = normalize_hmm_feature_sets(
                        getattr(cfg, "hmm_feature_sets", None)
                    )
                    feature_sets_access = {
                        k: v
                        for k, v in feature_sets_all.items()
                        if k in ("footprint", "accessible")
                    }
                    feature_sets_cpg = (
                        {"cpg": feature_sets_all["cpg"]} if "cpg" in feature_sets_all else {}
                    )

                    # =========================
                    # 1) Single-channel accessibility (per methbase)
                    # =========================
                    for mb in methbases:
                        try:
                            X, coords = build_single_channel(
                                subset,
                                ref=str(ref),
                                methbase=str(mb),
                                smf_modality=smf_modality,
                                cfg=cfg,
                            )
                        except Exception:
                            continue

                        arch = trainer.choose_arch(multichannel=False)
                        hmm = trainer.fit_or_load(
                            sample=str(sample),
                            ref=str(ref),
                            label=str(mb),
                            arch=arch,
                            X=X,
                            coords=coords,
                            device=device,
                        )

                        if not bypass_apply:
                            pm = _resolve_pos_mask_for_methbase(subset, str(ref), str(mb))
                            hmm.annotate_adata(
                                subset,
                                prefix=str(mb),
                                X=X,
                                coords=coords,
                                var_mask=pm,
                                span_fill=True,
                                config=cfg,
                                decode=decode,
                                write_posterior=write_post,
                                posterior_state=post_state,
                                feature_sets=feature_sets_access,  # <--- ONLY accessibility feature sets
                                prob_threshold=prob_thr,
                                uns_key=uns_key,
                                uns_flag=f"hmm_annotated_{mb}",
                                force_redo=force_apply,
                            )

                            # merges for this mb
                            for core_layer, dist in (
                                getattr(cfg, "hmm_merge_layer_features", []) or []
                            ):
                                base_layer = f"{mb}_{core_layer}"
                                if base_layer in subset.layers:
                                    merged_base = hmm.merge_intervals_to_new_layer(
                                        subset,
                                        base_layer,
                                        distance_threshold=int(dist),
                                        suffix=merged_suffix,
                                        overwrite=True,
                                    )
                                    # write merged size classes based on whichever group core_layer corresponds to
                                    for group, fs in feature_sets_access.items():
                                        fmap = fs.get("features", {}) or {}
                                        if fmap:
                                            hmm.write_size_class_layers_from_binary(
                                                subset,
                                                merged_base,
                                                out_prefix=str(mb),
                                                feature_ranges=fmap,
                                                suffix=merged_suffix,
                                                overwrite=True,
                                            )

                    # =========================
                    # 2) Multi-channel accessibility (Combined)
                    # =========================
                    if run_multi and len(methbases) >= 2:
                        try:
                            X3, coords_u, used_mbs = build_multi_channel_union(
                                subset,
                                ref=str(ref),
                                methbases=methbases,
                                smf_modality=smf_modality,
                                cfg=cfg,
                            )
                        except Exception:
                            X3, coords_u, used_mbs = None, None, []

                        if X3 is not None and len(used_mbs) >= 2:
                            union_mask = None
                            for mb in used_mbs:
                                pm = _resolve_pos_mask_for_methbase(subset, str(ref), str(mb))
                                union_mask = pm if union_mask is None else (union_mask | pm)

                            arch = trainer.choose_arch(multichannel=True)
                            hmmc = trainer.fit_or_load(
                                sample=str(sample),
                                ref=str(ref),
                                label="Combined",
                                arch=arch,
                                X=X3,
                                coords=coords_u,
                                device=device,
                            )

                            if not bypass_apply:
                                hmmc.annotate_adata(
                                    subset,
                                    prefix="Combined",
                                    X=X3,
                                    coords=coords_u,
                                    var_mask=union_mask,
                                    span_fill=True,
                                    config=cfg,
                                    decode=decode,
                                    write_posterior=write_post,
                                    posterior_state=post_state,
                                    feature_sets=feature_sets_access,  # <--- accessibility only
                                    prob_threshold=prob_thr,
                                    uns_key=uns_key,
                                    uns_flag="hmm_annotated_combined",
                                    force_redo=force_apply,
                                )

                                for core_layer, dist in (
                                    getattr(cfg, "hmm_merge_layer_features", []) or []
                                ):
                                    base_layer = f"Combined_{core_layer}"
                                    if base_layer in subset.layers:
                                        merged_base = hmmc.merge_intervals_to_new_layer(
                                            subset,
                                            base_layer,
                                            distance_threshold=int(dist),
                                            suffix=merged_suffix,
                                            overwrite=True,
                                        )
                                        for group, fs in feature_sets_access.items():
                                            fmap = fs.get("features", {}) or {}
                                            if fmap:
                                                hmmc.write_size_class_layers_from_binary(
                                                    subset,
                                                    merged_base,
                                                    out_prefix="Combined",
                                                    feature_ranges=fmap,
                                                    suffix=merged_suffix,
                                                    overwrite=True,
                                                )

                    # =========================
                    # 3) CpG-only single-channel task
                    # =========================
                    if run_cpg:
                        try:
                            Xcpg, coordscpg = build_single_channel(
                                subset,
                                ref=str(ref),
                                methbase="CpG",
                                smf_modality=smf_modality,
                                cfg=cfg,
                            )
                        except Exception:
                            Xcpg, coordscpg = None, None

                        if Xcpg is not None and Xcpg.size and feature_sets_cpg:
                            arch = trainer.choose_arch(multichannel=False)
                            hmmg = trainer.fit_or_load(
                                sample=str(sample),
                                ref=str(ref),
                                label="CpG",
                                arch=arch,
                                X=Xcpg,
                                coords=coordscpg,
                                device=device,
                            )

                            if not bypass_apply:
                                pm = _resolve_pos_mask_for_methbase(subset, str(ref), "CpG")
                                hmmg.annotate_adata(
                                    subset,
                                    prefix="CpG",
                                    X=Xcpg,
                                    coords=coordscpg,
                                    var_mask=pm,
                                    span_fill=True,
                                    config=cfg,
                                    decode=decode,
                                    write_posterior=write_post,
                                    posterior_state=post_state,
                                    feature_sets=feature_sets_cpg,  # <--- ONLY cpg group (cpg_patch)
                                    prob_threshold=prob_thr,
                                    uns_key=uns_key,
                                    uns_flag="hmm_annotated_CpG",
                                    force_redo=force_apply,
                                )

                    # ------------------------------------------------------------
                    # Copy newly created subset layers back into the full adata
                    # ------------------------------------------------------------
                    appended = (
                        list(subset.uns.get(uns_key, []))
                        if subset.uns.get(uns_key) is not None
                        else []
                    )
                    if appended:
                        row_mask = np.asarray(
                            mask.values if hasattr(mask, "values") else mask, dtype=bool
                        )

                        for ln in appended:
                            if ln not in subset.layers:
                                continue
                            _ensure_layer_and_assign_rows(adata, ln, row_mask, subset.layers[ln])
                            if ln not in global_appended:
                                global_appended.append(ln)

                        adata.uns[uns_key] = global_appended

            adata.uns["hmm_annotated"] = True

            hmm_layers = list(adata.uns.get("hmm_appended_layers", []) or [])
            # keep only real feature layers; drop lengths/states/posterior
            hmm_layers = [
                layer
                for layer in hmm_layers
                if not any(s in layer for s in ("_lengths", "_states", "_posterior"))
            ]
            logger.info(f"HMM appended layers: {hmm_layers}")

    # ---------------------------- HMM peak calling stage ----------------------------
    hmm_dir = pp_dir / "11_hmm_peak_calling"
    if hmm_dir.is_dir():
        pass
    else:
        make_dirs([pp_dir, hmm_dir])

        call_hmm_peaks(
            adata,
            feature_configs=cfg.hmm_peak_feature_configs,
            ref_column=cfg.reference_column,
            site_types=cfg.mod_target_bases,
            save_plot=True,
            output_dir=hmm_dir,
            index_col_suffix=cfg.reindexed_var_suffix,
        )

    ## Save HMM annotated adata
    if not paths.hmm.exists():
        logger.info("Saving hmm analyzed AnnData (post preprocessing and duplicate removal).")
        write_gz_h5ad(adata, paths.hmm)

    ########################################################################################################################

    ############################################### HMM based feature plotting ###############################################

    hmm_dir = pp_dir / "12_hmm_clustermaps"
    make_dirs([pp_dir, hmm_dir])

    layers: list[str] = []

    for base in cfg.hmm_methbases:
        layers.extend([f"{base}_{layer}" for layer in cfg.hmm_clustermap_feature_layers])

    if getattr(cfg, "hmm_run_multichannel", True) and len(cfg.hmm_methbases) >= 2:
        layers.extend([f"Combined_{layer}" for layer in cfg.hmm_clustermap_feature_layers])

    if cfg.cpg:
        layers.extend(["CpG_cpg_patch"])

    if not layers:
        raise ValueError(
            f"No HMM feature layers matched mod_target_bases={cfg.mod_target_bases} "
            f"and smf_modality={smf_modality}"
        )

    for layer in layers:
        hmm_cluster_save_dir = hmm_dir / layer
        if hmm_cluster_save_dir.is_dir():
            pass
        else:
            make_dirs([hmm_cluster_save_dir])

            combined_hmm_raw_clustermap(
                adata,
                sample_col=cfg.sample_name_col_for_plotting,
                reference_col=cfg.reference_column,
                hmm_feature_layer=layer,
                layer_gpc=cfg.layer_for_clustermap_plotting,
                layer_cpg=cfg.layer_for_clustermap_plotting,
                layer_c=cfg.layer_for_clustermap_plotting,
                layer_a=cfg.layer_for_clustermap_plotting,
                cmap_hmm=cfg.clustermap_cmap_hmm,
                cmap_gpc=cfg.clustermap_cmap_gpc,
                cmap_cpg=cfg.clustermap_cmap_cpg,
                cmap_c=cfg.clustermap_cmap_c,
                cmap_a=cfg.clustermap_cmap_a,
                min_quality=cfg.read_quality_filter_thresholds[0],
                min_length=cfg.read_len_filter_thresholds[0],
                min_mapped_length_to_reference_length_ratio=cfg.read_len_to_ref_ratio_filter_thresholds[
                    0
                ],
                min_position_valid_fraction=1 - cfg.position_max_nan_threshold,
                demux_types=("double", "already"),
                save_path=hmm_cluster_save_dir,
                normalize_hmm=False,
                sort_by=cfg.hmm_clustermap_sortby,  # options: 'gpc', 'cpg', 'gpc_cpg', 'none', or 'obs:<column>'
                bins=None,
                deaminase=deaminase,
                min_signal=0,
                index_col_suffix=cfg.reindexed_var_suffix,
            )

    hmm_dir = pp_dir / "13_hmm_bulk_traces"

    if hmm_dir.is_dir():
        logger.debug(f"{hmm_dir} already exists.")
    else:
        make_dirs([pp_dir, hmm_dir])
        from ..plotting import plot_hmm_layers_rolling_by_sample_ref

        bulk_hmm_layers = [
            layer
            for layer in hmm_layers
            if not any(s in layer for s in ("_lengths", "_states", "_posterior"))
        ]
        saved = plot_hmm_layers_rolling_by_sample_ref(
            adata,
            layers=bulk_hmm_layers,
            sample_col=cfg.sample_name_col_for_plotting,
            ref_col=cfg.reference_column,
            window=101,
            rows_per_page=4,
            figsize_per_cell=(4, 2.5),
            output_dir=hmm_dir,
            save=True,
            show_raw=False,
        )

    hmm_dir = pp_dir / "14_hmm_fragment_distributions"

    if hmm_dir.is_dir():
        logger.debug(f"{hmm_dir} already exists.")
    else:
        make_dirs([pp_dir, hmm_dir])
        from ..plotting import plot_hmm_size_contours

        if smf_modality == "deaminase":
            fragments = [
                ("C_all_accessible_features_lengths", 400),
                ("C_all_footprint_features_lengths", 250),
                ("C_all_accessible_features_merged_lengths", 800),
            ]
        elif smf_modality == "conversion":
            fragments = [
                ("GpC_all_accessible_features_lengths", 400),
                ("GpC_all_footprint_features_lengths", 250),
                ("GpC_all_accessible_features_merged_lengths", 800),
            ]
        elif smf_modality == "direct":
            fragments = [
                ("A_all_accessible_features_lengths", 400),
                ("A_all_footprint_features_lengths", 200),
                ("A_all_accessible_features_merged_lengths", 800),
            ]

        for layer, max in fragments:
            save_path = hmm_dir / layer
            make_dirs([save_path])

            figs = plot_hmm_size_contours(
                adata,
                length_layer=layer,
                sample_col=cfg.sample_name_col_for_plotting,
                ref_obs_col=cfg.reference_column,
                rows_per_page=6,
                max_length_cap=max,
                figsize_per_cell=(3.5, 2.2),
                save_path=save_path,
                save_pdf=False,
                save_each_page=True,
                dpi=200,
                smoothing_sigma=(10, 10),
                normalize_after_smoothing=True,
                cmap="Greens",
                log_scale_z=True,
            )
    ########################################################################################################################

    return (adata, paths.hmm)
