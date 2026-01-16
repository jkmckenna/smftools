from __future__ import annotations

from importlib import import_module

_LAZY_ATTRS = {
    "call_hmm_peaks": "smftools.hmm.call_hmm_peaks",
    "display_hmm": "smftools.hmm.display_hmm",
    "load_hmm": "smftools.hmm.hmm_readwrite",
    "save_hmm": "smftools.hmm.hmm_readwrite",
    "infer_nucleosomes_in_large_bound": "smftools.hmm.nucleosome_hmm_refinement",
    "refine_nucleosome_calls": "smftools.hmm.nucleosome_hmm_refinement",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "call_hmm_peaks",
    "display_hmm",
    "load_hmm",
    "refine_nucleosome_calls",
    "infer_nucleosomes_in_large_bound",
    "save_hmm",
]
