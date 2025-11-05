## readwrite ##

######################################################################################################
## Datetime functionality
def date_string():
    """
    Each time this is called, it returns the current date string
    """
    from datetime import datetime
    current_date = datetime.now()
    date_string = current_date.strftime("%Y%m%d")
    date_string = date_string[2:]
    return date_string

def time_string():
    """
    Each time this is called, it returns the current time string
    """
    from datetime import datetime
    current_time = datetime.now()
    return current_time.strftime("%H:%M:%S")
######################################################################################################

######################################################################################################
## Numpy, Pandas, Anndata functionality

def adata_to_df(adata, layer=None):
    """
    Convert an AnnData object into a Pandas DataFrame.

    Parameters:
        adata (AnnData): The input AnnData object.
        layer (str, optional): The layer to extract. If None, uses adata.X.

    Returns:
        pd.DataFrame: A DataFrame where rows are observations and columns are positions.
    """
    import pandas as pd
    import anndata as ad
    import numpy as np

    # Validate that the requested layer exists
    if layer and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")

    # Extract the data matrix
    data_matrix = adata.layers.get(layer, adata.X)

    # Ensure matrix is dense (handle sparse formats)
    if hasattr(data_matrix, "toarray"):  
        data_matrix = data_matrix.toarray()

    # Ensure obs and var have unique indices
    if adata.obs.index.duplicated().any():
        raise ValueError("Duplicate values found in `adata.obs.index`. Ensure unique observation indices.")
    
    if adata.var.index.duplicated().any():
        raise ValueError("Duplicate values found in `adata.var.index`. Ensure unique variable indices.")

    # Convert to DataFrame
    df = pd.DataFrame(data_matrix, index=adata.obs.index, columns=adata.var.index)

    return df


def save_matrix(matrix, save_name):
    """
    Input: A numpy matrix and a save_name
    Output: A txt file representation of the data matrix
    """
    import numpy as np
    np.savetxt(f'{save_name}.txt', matrix)

def concatenate_h5ads(output_file, file_suffix='h5ad.gz', delete_inputs=True):
    """
    Concatenate all h5ad files in a directory and delete them after the final adata is written out.
    Input: an output file path relative to the directory in which the function is called
    """
    import os
    import anndata as ad
    # Runtime warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='anndata')
    warnings.filterwarnings('ignore', category=FutureWarning, module='anndata')
    
    # List all files in the directory
    files = os.listdir(os.getcwd())
    # get current working directory
    cwd = os.getcwd()
    suffix = file_suffix
    # Filter file names that contain the search string in their filename and keep them in a list
    hdfs = [hdf for hdf in files if suffix in hdf]
    # Sort file list by names and print the list of file names
    hdfs.sort()
    print('{0} sample files found: {1}'.format(len(hdfs), hdfs))
    # Iterate over all of the hdf5 files and concatenate them.
    final_adata = None
    for hdf in hdfs:
        print('{0}: Reading in {1} hdf5 file'.format(time_string(), hdf))
        temp_adata = ad.read_h5ad(hdf)
        if final_adata:
            print('{0}: Concatenating final adata object with {1} hdf5 file'.format(time_string(), hdf))
            final_adata = ad.concat([final_adata, temp_adata], join='outer', index_unique=None)
        else:
            print('{0}: Initializing final adata object with {1} hdf5 file'.format(time_string(), hdf))
            final_adata = temp_adata
    print('{0}: Writing final concatenated hdf5 file'.format(time_string()))
    final_adata.write_h5ad(output_file, compression='gzip')

    # Delete the individual h5ad files and only keep the final concatenated file
    if delete_inputs:
        files = os.listdir(os.getcwd())
        hdfs = [hdf for hdf in files if suffix in hdf]
        if output_file in hdfs:
            hdfs.remove(output_file)
            # Iterate over the files and delete them
            for hdf in hdfs:
                try:
                    os.remove(hdf)
                    print(f"Deleted file: {hdf}")
                except OSError as e:
                    print(f"Error deleting file {hdf}: {e}")
    else:
        print('Keeping input files')

def safe_write_h5ad(adata, path, compression="gzip", backup=False, backup_dir="./uns_backups", verbose=True):
    """
    Save an AnnData safely by sanitizing .obs, .var, .uns, .layers, and .obsm.

    Returns a report dict and prints a summary of what was converted/backed up/skipped.
    """
    import os, json, pickle
    import numpy as np
    import pandas as pd
    import warnings
    import anndata as _ad

    os.makedirs(backup_dir, exist_ok=True)

    # report structure
    report = {
        "obs_converted_columns": [],
        "obs_backed_up_columns": [],
        "var_converted_columns": [],
        "var_backed_up_columns": [],
        "uns_backed_up_keys": [],
        "uns_json_keys": [],
        "layers_converted": [],
        "layers_skipped": [],
        "obsm_converted": [],
        "obsm_skipped": [],
        "X_replaced_or_converted": None,
        "errors": [],
    }

    def _backup(obj, name):
        """Pickle obj to backup_dir/name.pkl and return filename (or None)."""
        fname = os.path.join(backup_dir, f"{name}.pkl")
        try:
            with open(fname, "wb") as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"  backed up {name} -> {fname}")
            return fname
        except Exception as e:
            msg = f"failed to back up {name}: {e}"
            if verbose:
                print("  " + msg)
            report["errors"].append(msg)
            return None

    def _make_obs_var_safe(df: pd.DataFrame, which: str):
        """
        Return a sanitized copy of df where:
          - object columns converted to strings (with backup)
          - categorical columns' categories coerced to str (with backup)
        """
        df = df.copy()
        for col in list(df.columns):
            ser = df[col]
            # categorical handling
            try:
                is_cat = pd.api.types.is_categorical_dtype(ser.dtype)
            except Exception:
                is_cat = False

            if is_cat:
                try:
                    cats = ser.cat.categories
                    cats_str = cats.astype(str)
                    df[col] = pd.Categorical(ser.astype(str), categories=cats_str)
                    if verbose:
                        print(f"  coerced categorical column '{which}.{col}' -> string categories")
                    if which == "obs":
                        report["obs_converted_columns"].append(col)
                    else:
                        report["var_converted_columns"].append(col)
                except Exception:
                    # backup then coerce
                    if backup:
                        _backup(ser, f"{which}.{col}_categorical_backup")
                        if which == "obs":
                            report["obs_backed_up_columns"].append(col)
                        else:
                            report["var_backed_up_columns"].append(col)
                    df[col] = ser.astype(str)
                    if verbose:
                        print(f"  coerced categorical column '{which}.{col}' -> strings (backup={backup})")
                continue

            # object dtype handling: try to coerce each element to string
            try:
                is_obj = ser.dtype == object or pd.api.types.is_object_dtype(ser.dtype)
            except Exception:
                is_obj = False

            if is_obj:
                # test whether converting to string succeeds for all elements
                try:
                    _ = np.array(ser.values.astype(str))
                    if backup:
                        _backup(ser.values, f"{which}.{col}_backup")
                        if which == "obs":
                            report["obs_backed_up_columns"].append(col)
                        else:
                            report["var_backed_up_columns"].append(col)
                    df[col] = ser.values.astype(str)
                    if verbose:
                        print(f"  converted object column '{which}.{col}' -> strings (backup={backup})")
                    if which == "obs":
                        report["obs_converted_columns"].append(col)
                    else:
                        report["var_converted_columns"].append(col)
                except Exception:
                    # fallback: attempt per-element json.dumps; if fails mark as backed-up and coerce via str()
                    convertible = True
                    for val in ser.values:
                        try:
                            json.dumps(val, default=str)
                        except Exception:
                            convertible = False
                            break
                    if convertible:
                        if backup:
                            _backup(ser.values, f"{which}.{col}_backup")
                            if which == "obs":
                                report["obs_backed_up_columns"].append(col)
                            else:
                                report["var_backed_up_columns"].append(col)
                        df[col] = [json.dumps(v, default=str) for v in ser.values]
                        if verbose:
                            print(f"  json-stringified object column '{which}.{col}' (backup={backup})")
                        if which == "obs":
                            report["obs_converted_columns"].append(col)
                        else:
                            report["var_converted_columns"].append(col)
                    else:
                        # fallback to string repr and backup
                        if backup:
                            _backup(ser.values, f"{which}.{col}_backup")
                            if which == "obs":
                                report["obs_backed_up_columns"].append(col)
                            else:
                                report["var_backed_up_columns"].append(col)
                        df[col] = ser.astype(str)
                        if verbose:
                            print(f"  WARNING: column '{which}.{col}' was complex; coerced via str() (backed up).")
                        if which == "obs":
                            report["obs_converted_columns"].append(col)
                        else:
                            report["var_converted_columns"].append(col)
        return df

    def _sanitize_uns(uns: dict):
        """
        For each key/value in uns:
          - if json.dumps(value) works: keep it
          - else: pickle value to backup dir, and add a JSON-stringified representation under key+'_json'
        """
        clean = {}
        backed_up = []
        for k, v in uns.items():
            try:
                json.dumps(v)
                clean[k] = v
            except Exception:
                try:
                    s = json.dumps(v, default=str)
                    clean[k + "_json"] = s
                    if backup:
                        _backup(v, f"uns_{k}_backup")
                    backed_up.append(k)
                    if verbose:
                        print(f"  uns['{k}'] non-JSON -> stored '{k}_json' and backed up (backup={backup})")
                    report["uns_json_keys"].append(k)
                except Exception:
                    try:
                        if backup:
                            _backup(v, f"uns_{k}_backup")
                        clean[k + "_str"] = str(v)
                        backed_up.append(k)
                        if verbose:
                            print(f"  uns['{k}'] stored as string under '{k}_str' (backed up).")
                        report["uns_backed_up_keys"].append(k)
                    except Exception as e:
                        msg = f"uns['{k}'] could not be preserved: {e}"
                        report["errors"].append(msg)
                        if verbose:
                            print("  " + msg)
        if backed_up and verbose:
            print(f"Sanitized .uns keys (backed up): {backed_up}")
        return clean

    def _sanitize_layers_obsm(src_dict, which: str):
        """
        Ensure arrays in layers/obsm are numeric and non-object dtype.
        Returns a cleaned dict suitable to pass into AnnData(...)
        If an entry is not convertible, it is backed up & skipped.
        """
        cleaned = {}
        for k, v in src_dict.items():
            try:
                arr = np.asarray(v)
                if arr.dtype == object:
                    try:
                        arr_f = arr.astype(float)
                        cleaned[k] = arr_f
                        report_key = f"{which}.{k}"
                        report["layers_converted"].append(report_key) if which == "layers" else report["obsm_converted"].append(report_key)
                        if verbose:
                            print(f"  {which}.{k} object array coerced to float.")
                    except Exception:
                        try:
                            arr_i = arr.astype(int)
                            cleaned[k] = arr_i
                            report_key = f"{which}.{k}"
                            report["layers_converted"].append(report_key) if which == "layers" else report["obsm_converted"].append(report_key)
                            if verbose:
                                print(f"  {which}.{k} object array coerced to int.")
                        except Exception:
                            if backup:
                                _backup(v, f"{which}_{k}_backup")
                            if which == "layers":
                                report["layers_skipped"].append(k)
                            else:
                                report["obsm_skipped"].append(k)
                            if verbose:
                                print(f"  SKIPPING {which}.{k} (object dtype not numeric). Backed up: {backup}")
                            continue
                else:
                    cleaned[k] = arr
            except Exception as e:
                if backup:
                    _backup(v, f"{which}_{k}_backup")
                if which == "layers":
                    report["layers_skipped"].append(k)
                else:
                    report["obsm_skipped"].append(k)
                msg = f"  SKIPPING {which}.{k} due to conversion error: {e}"
                report["errors"].append(msg)
                if verbose:
                    print(msg)
                continue
        return cleaned

    # ---------- sanitize obs, var ----------
    try:
        obs_clean = _make_obs_var_safe(adata.obs, "obs")
    except Exception as e:
        msg = f"Failed to sanitize obs: {e}"
        report["errors"].append(msg)
        if verbose:
            print(msg)
        obs_clean = adata.obs.copy()

    try:
        var_clean = _make_obs_var_safe(adata.var, "var")
    except Exception as e:
        msg = f"Failed to sanitize var: {e}"
        report["errors"].append(msg)
        if verbose:
            print(msg)
        var_clean = adata.var.copy()

    # ---------- sanitize uns ----------
    try:
        uns_clean = _sanitize_uns(adata.uns)
    except Exception as e:
        msg = f"Failed to sanitize uns: {e}"
        report["errors"].append(msg)
        if verbose:
            print(msg)
        uns_clean = {}

    # ---------- sanitize layers and obsm ----------
    layers_src = getattr(adata, "layers", {})
    obsm_src = getattr(adata, "obsm", {})

    try:
        layers_clean = _sanitize_layers_obsm(layers_src, "layers")
    except Exception as e:
        msg = f"Failed to sanitize layers: {e}"
        report["errors"].append(msg)
        if verbose:
            print(msg)
        layers_clean = {}

    try:
        obsm_clean = _sanitize_layers_obsm(obsm_src, "obsm")
    except Exception as e:
        msg = f"Failed to sanitize obsm: {e}"
        report["errors"].append(msg)
        if verbose:
            print(msg)
        obsm_clean = {}

    # ---------- handle X ----------
    X_to_use = adata.X
    try:
        X_arr = np.asarray(adata.X)
        if X_arr.dtype == object:
            try:
                X_to_use = X_arr.astype(float)
                report["X_replaced_or_converted"] = "converted_to_float"
                if verbose:
                    print("Converted adata.X object-dtype -> float")
            except Exception:
                if backup:
                    _backup(adata.X, "X_backup")
                X_to_use = np.zeros_like(X_arr, dtype=float)
                report["X_replaced_or_converted"] = "replaced_with_zeros_backup"
                if verbose:
                    print("adata.X had object dtype and couldn't be converted; replaced with zeros (backup set).")
    except Exception as e:
        msg = f"Error handling adata.X: {e}"
        report["errors"].append(msg)
        if verbose:
            print(msg)
        X_to_use = adata.X

    # ---------- build lightweight AnnData copy ----------
    try:
        adata_copy = _ad.AnnData(
            X=X_to_use,
            obs=obs_clean,
            var=var_clean,
            layers=layers_clean,
            uns=uns_clean,
            obsm=obsm_clean,
            varm=getattr(adata, "varm", None),
        )

        # preserve names (as strings)
        try:
            adata_copy.obs_names = adata.obs_names.astype(str)
            adata_copy.var_names = adata.var_names.astype(str)
        except Exception:
            adata_copy.obs_names = adata.obs_names
            adata_copy.var_names = adata.var_names

        # --- write
        adata_copy.write_h5ad(path, compression=compression)
        if verbose:
            print(f"Saved safely to {path}")
    except Exception as e:
        msg = f"Failed to write h5ad: {e}"
        report["errors"].append(msg)
        if verbose:
            print(msg)
        raise

    # Print a concise interactive report
    print("\n=== safe_write_h5ad REPORT ===")
    print(f"Saved file: {path}")
    print(f"Adata shape: {adata.shape}")
    if report["obs_converted_columns"] or report["obs_backed_up_columns"]:
        print(f"obs: converted columns -> {report['obs_converted_columns']}")
        print(f"obs: backed-up columns -> {report['obs_backed_up_columns']}")
    else:
        print("obs: no problematic columns found.")

    if report["var_converted_columns"] or report["var_backed_up_columns"]:
        print(f"var: converted columns -> {report['var_converted_columns']}")
        print(f"var: backed-up columns -> {report['var_backed_up_columns']}")
    else:
        print("var: no problematic columns found.")

    if report["uns_json_keys"] or report["uns_backed_up_keys"]:
        print(f".uns: jsonified keys -> {report['uns_json_keys']}")
        print(f".uns: backed-up keys -> {report['uns_backed_up_keys']}")
    else:
        print(".uns: no problematic keys found.")

    if report["layers_converted"] or report["layers_skipped"]:
        print(f"layers: converted -> {report['layers_converted']}")
        print(f"layers: skipped -> {report['layers_skipped']}")
    else:
        print("layers: no problematic entries found.")

    if report["obsm_converted"] or report["obsm_skipped"]:
        print(f"obsm: converted -> {report['obsm_converted']}")
        print(f"obsm: skipped -> {report['obsm_skipped']}")
    else:
        print("obsm: no problematic entries found.")

    if report["X_replaced_or_converted"]:
        print(f"adata.X handled: {report['X_replaced_or_converted']}")
    else:
        print("adata.X: no changes.")

    if report["errors"]:
        print("\nWarnings / errors encountered:")
        for e in report["errors"]:
            print(" -", e)

    print("=== end report ===\n")
    return report

def safe_read_h5ad(path, backup_dir="./uns_backups", restore_backups=True, re_categorize=True, categorical_threshold=100, verbose=True):
    """
    Safely load an AnnData saved by safe_write_h5ad and attempt to restore complex objects
    from the backup_dir produced during save.

    Parameters
    ----------
    path : str
        Path to the cleaned .h5ad produced by safe_write_h5ad.
    backup_dir : str
        Directory where safe_write_h5ad stored pickled backups (default "./uns_backups").
    restore_backups : bool
        If True, attempt to load pickled backups and restore original objects into adata.
    re_categorize : bool
        If True, try to coerce small unique-count string columns back into pandas.Categorical.
    categorical_threshold : int
        Max unique values for a column to be considered categorical for automatic recasting.
    verbose : bool
        Print progress/summary.

    Returns
    -------
    (adata, report) :
        adata : AnnData
            The reloaded (and possibly restored) AnnData instance.
        report : dict
            A report describing restored items, parsed JSON keys, and any failures.
    """
    import os
    import json
    import pickle
    import numpy as np
    import pandas as pd
    import anndata as _ad

    report = {
        "restored_obs_columns": [],
        "restored_var_columns": [],
        "restored_uns_keys": [],
        "parsed_uns_json_keys": [],
        "restored_layers": [],
        "restored_obsm": [],
        "recategorized_obs": [],
        "recategorized_var": [],
        "missing_backups": [],
        "errors": [],
    }

    if verbose:
        print(f"[safe_read_h5ad] loading {path}")

    # 1) load the cleaned h5ad
    try:
        adata = _ad.read_h5ad(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read h5ad at {path}: {e}")

    # Ensure backup_dir exists (may be relative to cwd)
    backup_dir = os.path.abspath(backup_dir)
    if verbose:
        print(f"[safe_read_h5ad] looking for backups in {backup_dir}")

    def _load_pickle_if_exists(fname):
        if os.path.exists(fname):
            try:
                with open(fname, "rb") as fh:
                    val = pickle.load(fh)
                return val
            except Exception as e:
                report["errors"].append(f"Failed to load pickle {fname}: {e}")
                if verbose:
                    print(f"  error loading {fname}: {e}")
                return None
        return None

    # 2) Restore obs columns
    for col in list(adata.obs.columns):
        # Look for backup with exact naming from safe_write_h5ad: "obs.<col>_backup.pkl" or "obs.<col>_categorical_backup.pkl"
        bname1 = os.path.join(backup_dir, f"obs.{col}_backup.pkl")
        bname2 = os.path.join(backup_dir, f"obs.{col}_categorical_backup.pkl")
        restored = False

        if restore_backups:
            val = _load_pickle_if_exists(bname2)
            if val is not None:
                # val may be the categorical series or categories
                try:
                    # If pickled numpy array or pandas Series, coerce to same index alignment
                    if hasattr(val, "shape") and (len(val) == adata.shape[0]):
                        adata.obs[col] = pd.Series(val, index=adata.obs.index)
                    else:
                        # fallback: place pickled object directly
                        adata.obs[col] = pd.Series([val] * adata.shape[0], index=adata.obs.index)
                    report["restored_obs_columns"].append((col, bname2))
                    restored = True
                    if verbose:
                        print(f"[safe_read_h5ad] restored obs.{col} from {bname2}")
                except Exception as e:
                    report["errors"].append(f"Failed to restore obs.{col} from {bname2}: {e}")
                    restored = False

            if not restored:
                val = _load_pickle_if_exists(bname1)
                if val is not None:
                    try:
                        if hasattr(val, "shape") and (len(val) == adata.shape[0]):
                            adata.obs[col] = pd.Series(val, index=adata.obs.index)
                        else:
                            adata.obs[col] = pd.Series([val] * adata.shape[0], index=adata.obs.index)
                        report["restored_obs_columns"].append((col, bname1))
                        restored = True
                        if verbose:
                            print(f"[safe_read_h5ad] restored obs.{col} from {bname1}")
                    except Exception as e:
                        report["errors"].append(f"Failed to restore obs.{col} from {bname1}: {e}")
                        restored = False

        # If not restored and column dtype is object but contains JSON-like strings, try json.loads per element
        if (not restored) and (adata.obs[col].dtype == object):
            sample_vals = adata.obs[col].dropna().astype(str).head(20).tolist()
            looks_like_json = False
            for sv in sample_vals:
                svs = sv.strip()
                if (svs.startswith("{") and svs.endswith("}")) or (svs.startswith("[") and svs.endswith("]")):
                    looks_like_json = True
                    break
            if looks_like_json:
                parsed = []
                success_parse = True
                for v in adata.obs[col].astype(str).values:
                    try:
                        parsed.append(json.loads(v))
                    except Exception:
                        # if any element fails, don't convert whole column
                        success_parse = False
                        break
                if success_parse:
                    adata.obs[col] = pd.Series(parsed, index=adata.obs.index)
                    report["restored_obs_columns"].append((col, "parsed_json"))
                    restored = True
                    if verbose:
                        print(f"[safe_read_h5ad] parsed obs.{col} JSON strings back to Python objects")

        # If still not restored and re_categorize=True, try to convert small unique string columns back to categorical
        if (not restored) and re_categorize and adata.obs[col].dtype == object:
            try:
                nunique = adata.obs[col].dropna().astype(str).nunique()
                if nunique > 0 and nunique <= categorical_threshold:
                    # cast to category
                    adata.obs[col] = adata.obs[col].astype(str).astype("category")
                    report["recategorized_obs"].append(col)
                    if verbose:
                        print(f"[safe_read_h5ad] recast obs.{col} -> categorical (n_unique={nunique})")
            except Exception as e:
                report["errors"].append(f"Failed to recategorize obs.{col}: {e}")

    # 3) Restore var columns (same logic)
    for col in list(adata.var.columns):
        bname1 = os.path.join(backup_dir, f"var.{col}_backup.pkl")
        bname2 = os.path.join(backup_dir, f"var.{col}_categorical_backup.pkl")
        restored = False

        if restore_backups:
            val = _load_pickle_if_exists(bname2)
            if val is not None:
                try:
                    if hasattr(val, "shape") and (len(val) == adata.shape[1]):
                        adata.var[col] = pd.Series(val, index=adata.var.index)
                    else:
                        adata.var[col] = pd.Series([val] * adata.shape[1], index=adata.var.index)
                    report["restored_var_columns"].append((col, bname2))
                    restored = True
                    if verbose:
                        print(f"[safe_read_h5ad] restored var.{col} from {bname2}")
                except Exception as e:
                    report["errors"].append(f"Failed to restore var.{col} from {bname2}: {e}")

            if not restored:
                val = _load_pickle_if_exists(bname1)
                if val is not None:
                    try:
                        if hasattr(val, "shape") and (len(val) == adata.shape[1]):
                            adata.var[col] = pd.Series(val, index=adata.var.index)
                        else:
                            adata.var[col] = pd.Series([val] * adata.shape[1], index=adata.var.index)
                        report["restored_var_columns"].append((col, bname1))
                        restored = True
                        if verbose:
                            print(f"[safe_read_h5ad] restored var.{col} from {bname1}")
                    except Exception as e:
                        report["errors"].append(f"Failed to restore var.{col} from {bname1}: {e}")

        if (not restored) and (adata.var[col].dtype == object):
            # try JSON parsing
            sample_vals = adata.var[col].dropna().astype(str).head(20).tolist()
            looks_like_json = False
            for sv in sample_vals:
                svs = sv.strip()
                if (svs.startswith("{") and svs.endswith("}")) or (svs.startswith("[") and svs.endswith("]")):
                    looks_like_json = True
                    break
            if looks_like_json:
                parsed = []
                success_parse = True
                for v in adata.var[col].astype(str).values:
                    try:
                        parsed.append(json.loads(v))
                    except Exception:
                        success_parse = False
                        break
                if success_parse:
                    adata.var[col] = pd.Series(parsed, index=adata.var.index)
                    report["restored_var_columns"].append((col, "parsed_json"))
                    if verbose:
                        print(f"[safe_read_h5ad] parsed var.{col} JSON strings back to Python objects")

        if (not restored) and re_categorize and adata.var[col].dtype == object:
            try:
                nunique = adata.var[col].dropna().astype(str).nunique()
                if nunique > 0 and nunique <= categorical_threshold:
                    adata.var[col] = adata.var[col].astype(str).astype("category")
                    report["recategorized_var"].append(col)
                    if verbose:
                        print(f"[safe_read_h5ad] recast var.{col} -> categorical (n_unique={nunique})")
            except Exception as e:
                report["errors"].append(f"Failed to recategorize var.{col}: {e}")

    # 4) Restore uns: look for uns_{k}_backup.pkl, or keys like "<k>_json"
    uns_keys = list(adata.uns.keys())
    # First, if we have "<k>_json", convert back into k
    for k in uns_keys:
        if k.endswith("_json"):
            base = k[:-5]
            sval = adata.uns.get(k)
            try:
                parsed = json.loads(sval)
                adata.uns[base] = parsed
                report["parsed_uns_json_keys"].append(base)
                if verbose:
                    print(f"[safe_read_h5ad] parsed adata.uns['{k}'] -> adata.uns['{base}']")
                # remove the _json entry
                try:
                    del adata.uns[k]
                except KeyError:
                    pass
            except Exception as e:
                report["errors"].append(f"Failed to json-parse uns['{k}']: {e}")

    # Now try to restore pickled backups for uns keys
    # Look for files named uns_<key>_backup.pkl
    # We will attempt to restore into adata.uns[key] if backup exists
    for fname in os.listdir(backup_dir) if os.path.isdir(backup_dir) else []:
        if not fname.startswith("uns_") or not fname.endswith("_backup.pkl"):
            continue
        # fname example: "uns_clustermap_results_backup.pkl" -> key name between 'uns_' and '_backup.pkl'
        key = fname[len("uns_"):-len("_backup.pkl")]
        full = os.path.join(backup_dir, fname)
        val = _load_pickle_if_exists(full)
        if val is not None:
            adata.uns[key] = val
            report["restored_uns_keys"].append((key, full))
            if verbose:
                print(f"[safe_read_h5ad] restored adata.uns['{key}'] from {full}")

    # 5) Restore layers and obsm from backups if present
    # expected backup names: layers_<name>_backup.pkl, obsm_<name>_backup.pkl
    if os.path.isdir(backup_dir):
        for fname in os.listdir(backup_dir):
            if fname.startswith("layers_") and fname.endswith("_backup.pkl"):
                layer_name = fname[len("layers_"):-len("_backup.pkl")]
                full = os.path.join(backup_dir, fname)
                val = _load_pickle_if_exists(full)
                if val is not None:
                    try:
                        adata.layers[layer_name] = np.asarray(val)
                        report["restored_layers"].append((layer_name, full))
                        if verbose:
                            print(f"[safe_read_h5ad] restored layers['{layer_name}'] from {full}")
                    except Exception as e:
                        report["errors"].append(f"Failed to restore layers['{layer_name}'] from {full}: {e}")

            if fname.startswith("obsm_") and fname.endswith("_backup.pkl"):
                obsm_name = fname[len("obsm_"):-len("_backup.pkl")]
                full = os.path.join(backup_dir, fname)
                val = _load_pickle_if_exists(full)
                if val is not None:
                    try:
                        adata.obsm[obsm_name] = np.asarray(val)
                        report["restored_obsm"].append((obsm_name, full))
                        if verbose:
                            print(f"[safe_read_h5ad] restored obsm['{obsm_name}'] from {full}")
                    except Exception as e:
                        report["errors"].append(f"Failed to restore obsm['{obsm_name}'] from {full}: {e}")

    # 6) If restore_backups True but some expected backups missing, note them
    if restore_backups and os.path.isdir(backup_dir):
        # detect common expected names from obs/var/uns/layers in adata
        expected_missing = []
        # obs/var columns
        for col in list(adata.obs.columns):
            p1 = os.path.join(backup_dir, f"obs.{col}_backup.pkl")
            p2 = os.path.join(backup_dir, f"obs.{col}_categorical_backup.pkl")
            if (not os.path.exists(p1)) and (not os.path.exists(p2)):
                # we don't require backups for every column; only record if column still looks like placeholder strings
                if adata.obs[col].dtype == object:
                    expected_missing.append(("obs", col))
        for col in list(adata.var.columns):
            p1 = os.path.join(backup_dir, f"var.{col}_backup.pkl")
            p2 = os.path.join(backup_dir, f"var.{col}_categorical_backup.pkl")
            if (not os.path.exists(p1)) and (not os.path.exists(p2)):
                if adata.var[col].dtype == object:
                    expected_missing.append(("var", col))
        # uns keys
        for k in adata.uns.keys():
            # if we have *_json or *_str variants we expect backups optionally
            if k.endswith("_json") or k.endswith("_str"):
                b = os.path.join(backup_dir, f"uns_{k[:-5]}_backup.pkl")
                if not os.path.exists(b):
                    report["missing_backups"].append(("uns", k))
        if expected_missing and verbose:
            n = len(expected_missing)
            if verbose:
                print(f"[safe_read_h5ad] note: {n} obs/var object columns may not have backups; check if their content is acceptable.")
            # add to report
            report["missing_backups"].extend(expected_missing)

    # final summary print
    if verbose:
        print("\n=== safe_read_h5ad summary ===")
        if report["restored_obs_columns"]:
            print("Restored obs columns:", report["restored_obs_columns"])
        if report["restored_var_columns"]:
            print("Restored var columns:", report["restored_var_columns"])
        if report["restored_uns_keys"]:
            print("Restored uns keys:", report["restored_uns_keys"])
        if report["parsed_uns_json_keys"]:
            print("Parsed uns JSON keys:", report["parsed_uns_json_keys"])
        if report["restored_layers"]:
            print("Restored layers:", report["restored_layers"])
        if report["restored_obsm"]:
            print("Restored obsm:", report["restored_obsm"])
        if report["recategorized_obs"] or report["recategorized_var"]:
            print("Recategorized columns (obs/var):", report["recategorized_obs"], report["recategorized_var"])
        if report["missing_backups"]:
            print("Missing backups or object columns without backups (investigate):", report["missing_backups"])
        if report["errors"]:
            print("Errors encountered (see report['errors']):")
            for e in report["errors"]:
                print(" -", e)
        print("=== end summary ===\n")

    return adata, report


# def safe_write_h5ad(adata, path, compression="gzip", backup=False, backup_dir="./", verbose=True):
#     """
#     Saves an AnnData object safely by omitting problematic columns from .obs and .var.

#     Parameters:
#         adata (AnnData): The AnnData object to save.
#         path (str): Output .h5ad file path.
#         compression (str): Compression method for h5ad file.
#         backup (bool): If True, saves problematic columns to CSV files.
#         backup_dir (str): Directory to store backups if backup=True.
#     """
#     import anndata as ad
#     import pandas as pd
#     import os
#     import numpy as np
#     import json

#     os.makedirs(backup_dir, exist_ok=True)

#     def filter_df(df, df_name):
#         bad_cols = []
#         for col in df.columns:
#             if df[col].dtype == 'object':
#                 if not df[col].apply(lambda x: isinstance(x, (str, type(None)))).all():
#                     bad_cols.append(col)
#             elif pd.api.types.is_categorical_dtype(df[col]):
#                 if not all(isinstance(x, (str, type(None))) for x in df[col].cat.categories):
#                     bad_cols.append(col)
#         if bad_cols and verbose:
#             print(f"Skipping columns from {df_name}: {bad_cols}")
#         if backup and bad_cols:
#             df[bad_cols].to_csv(os.path.join(backup_dir, f"{df_name}_skipped_columns.csv"))
#             if verbose:
#                 print(f"Backed up skipped columns to {backup_dir}/{df_name}_skipped_columns.csv")
#         return df.drop(columns=bad_cols)
    
#     def is_serializable(val):
#         try:
#             json.dumps(val)
#             return True
#         except (TypeError, OverflowError):
#             return False

#     def clean_uns(uns_dict):
#         clean_uns = {}
#         bad_keys = []
#         for k, v in uns_dict.items():
#             if isinstance(v, (str, int, float, type(None), list, np.ndarray, pd.DataFrame, dict)):
#                 clean_uns[k] = v
#             elif is_serializable(v):
#                 clean_uns[k] = v
#             else:
#                 bad_keys.append(k)
#                 if backup:
#                     try:
#                         with open(os.path.join(backup_dir, f"uns_{k}_backup.txt"), "w") as f:
#                             f.write(str(v))
#                     except Exception:
#                         pass
#         if bad_keys and verbose:
#             print(f"Skipping entries from .uns: {bad_keys}")
#         return clean_uns

#     # Clean obs and var and uns
#     obs_clean = filter_df(adata.obs, "obs")
#     var_clean = filter_df(adata.var, "var")
#     uns_clean = clean_uns(adata.uns)

#     # Save clean version
#     adata_copy = ad.AnnData(
#         X=adata.X,
#         obs=obs_clean,
#         var=var_clean,
#         layers=adata.layers,
#         uns=uns_clean,
#         obsm=adata.obsm,
#         varm=adata.varm
#     )

#     adata_copy.obs_names = adata_copy.obs_names.astype(str)
#     adata_copy.var_names = adata_copy.var_names.astype(str)

#     adata_copy.write_h5ad(path, compression=compression)

#     print(f"Saved safely to {path}")

def merge_barcoded_anndatas_core(adata_single, adata_double):
    import numpy as np
    import anndata as ad

    # Step 1: Identify overlap
    overlap = np.intersect1d(adata_single.obs_names, adata_double.obs_names)

    # Step 2: Filter out overlaps from adata_single
    adata_single_filtered = adata_single[~adata_single.obs_names.isin(overlap)].copy()

    # Step 3: Add source tag
    adata_single_filtered.obs['source'] = 'single_barcode'
    adata_double.obs['source'] = 'double_barcode'

    # Step 4: Concatenate all components
    adata_merged = ad.concat([
        adata_single_filtered,
        adata_double
    ], join='outer', merge='same')  # merge='same' preserves matching layers, obsm, etc.

    # Step 5: Merge `.uns`
    adata_merged.uns = {**adata_single.uns, **adata_double.uns}

    return adata_merged
######################################################################################################

### File conversion misc ###
import argparse
from Bio import SeqIO
def genbank_to_gff(genbank_file, output_file, record_id):
    with open(output_file, "w") as out:
        for record in SeqIO.parse(genbank_file, "genbank"):
            for feature in record.features:
                # Skip features without location information
                if feature.location is None:
                    continue
                # Extract feature information
                start = feature.location.start + 1  # Convert to 1-based index
                end = feature.location.end
                strand = "+" if feature.location.strand == 1 else "-"
                feature_type = feature.type
                # Format attributes
                attributes = ";".join(f"{k}={v}" for k, v in feature.qualifiers.items())
                # Write GFF3 line
                gff3_line = "\t".join(str(x) for x in [record_id, feature.type, feature_type, start, end, ".", strand, ".", attributes])
                out.write(gff3_line + "\n")