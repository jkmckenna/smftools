from pathlib import Path
import pandas as pd
import numpy as np

def add_demux_type_annotation(
    adata,
    double_demux_source,
    sep: str = "\t",
    read_id_col: str = "read_id",
    barcode_col: str = "barcode",
):
    """
    Add adata.obs["demux_type"]:
        - "double" if read_id appears in the *double demux* TSV
        - "single" otherwise

    Rows where barcode == "unclassified" in the demux TSV are ignored.

    Parameters
    ----------
    adata : AnnData
        AnnData object whose obs_names are read_ids.
    double_demux_source : str | Path | list[str]
        Either:
          - path to a TSV/TXT of dorado demux results
          - a list of read_ids
    """

    # -----------------------------
    # If it's a file → load TSV
    # -----------------------------
    if isinstance(double_demux_source, (str, Path)):
        file_path = Path(double_demux_source)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        df = pd.read_csv(file_path, sep=sep, dtype=str)

        # If the file has only one column → treat as a simple read list
        if df.shape[1] == 1:
            read_ids = df.iloc[:, 0].tolist()
        else:
            # Validate columns
            if read_id_col not in df.columns:
                raise ValueError(f"TSV must contain a '{read_id_col}' column.")
            if barcode_col not in df.columns:
                raise ValueError(f"TSV must contain a '{barcode_col}' column.")

            # Drop unclassified reads
            df = df[df[barcode_col].str.lower() != "unclassified"]

            # Extract read_ids
            read_ids = df[read_id_col].tolist()

    # -----------------------------
    # If user supplied list-of-ids
    # -----------------------------
    else:
        read_ids = list(double_demux_source)

    # Deduplicate for speed
    double_set = set(read_ids)

    # Boolean lookup in AnnData
    is_double = adata.obs_names.isin(double_set)

    adata.obs["demux_type"] = np.where(is_double, "double", "single")
    adata.obs["demux_type"] = adata.obs["demux_type"].astype("category")

    return adata
