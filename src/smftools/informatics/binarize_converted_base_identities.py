from __future__ import annotations


def binarize_converted_base_identities(
    base_identities,
    strand,
    modification_type,
    deaminase_footprinting=False,
    mismatch_trend_per_read={},
    on_missing="nan",
):
    """
    Efficiently binarizes conversion SMF data within a sequence string using NumPy arrays.
    For conversion modality, the strand parameter is used for mapping.
    For deaminase modality, the mismatch_trend_per_read is used for mapping.

    Parameters:
        base_identities (dict): A dictionary returned by extract_base_identities. Keyed by read name. Points to a list of base identities.
        strand (str): A string indicating which strand was converted in the experiment (options are 'top' and 'bottom').
        modification_type (str): A string indicating the modification type of interest (options are '5mC' and '6mA').
        deaminase_footprinting (bool): Whether direct deaminase footprinting chemistry was used.
        mismatch_trend_per_read (dict): For deaminase footprinting, indicates the type of conversion relative to the top strand reference for each read. (C->T or G->A if bottom strand was converted)
        on_missing (str): Error handling if a read is missing

    Returns:
        dict: A dictionary where 1 represents a methylated site, 0 represents an unmethylated site, and NaN represents a site without methylation info.
        If deaminase_footprinting, 1 represents deaminated sites, while 0 represents non-deaminated sites.
    """
    import numpy as np

    if mismatch_trend_per_read is None:
        mismatch_trend_per_read = {}

    # Fast path
    if modification_type == "unconverted" and not deaminase_footprinting:
        return {k: np.full(len(v), np.nan, dtype=np.float32) for k, v in base_identities.items()}

    out = {}

    if deaminase_footprinting:
        valid_trends = {"C->T", "G->A"}

        for read_id, bases in base_identities.items():
            trend_raw = mismatch_trend_per_read.get(read_id, None)
            if trend_raw is None:
                if on_missing == "error":
                    raise KeyError(f"Missing mismatch trend for read '{read_id}'")
                out[read_id] = np.full(len(bases), np.nan, dtype=np.float32)
                continue

            trend = trend_raw.replace(" ", "").upper()
            if trend not in valid_trends:
                if on_missing == "error":
                    raise KeyError(
                        f"Invalid mismatch trend '{trend_raw}' for read '{read_id}'. "
                        f"Expected one of {sorted(valid_trends)}"
                    )
                out[read_id] = np.full(len(bases), np.nan, dtype=np.float32)
                continue

            arr = np.asarray(bases, dtype="<U1")
            res = np.full(arr.shape, np.nan, dtype=np.float32)

            if trend == "C->T":
                # C (unconverted) -> 0, T (converted) -> 1
                res[arr == "C"] = 0.0
                res[arr == "T"] = 1.0
            else:  # "G->A"
                res[arr == "G"] = 0.0
                res[arr == "A"] = 1.0

            out[read_id] = res

        return out

    # Non-deaminase mapping (bisulfite-style for 5mC; 6mA mapping is protocol dependent)
    bin_maps = {
        ("top", "5mC"): {"C": 1.0, "T": 0.0},
        ("bottom", "5mC"): {"G": 1.0, "A": 0.0},
        ("top", "6mA"): {"A": 1.0, "G": 0.0},
        ("bottom", "6mA"): {"T": 1.0, "C": 0.0},
    }
    key = (strand, modification_type)
    if key not in bin_maps:
        raise ValueError(
            f"Invalid combination of strand='{strand}' and modification_type='{modification_type}'"
        )

    base_map = bin_maps[key]

    for read_id, bases in base_identities.items():
        arr = np.asarray(bases, dtype="<U1")
        res = np.full(arr.shape, np.nan, dtype=np.float32)
        # mask-assign; unknown characters (N, -, etc.) remain NaN
        for b, v in base_map.items():
            res[arr == b] = v
        out[read_id] = res

    return out
