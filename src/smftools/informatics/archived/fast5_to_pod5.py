from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Union, List

def fast5_to_pod5(
    fast5_dir: Union[str, Path, List[Union[str, Path]]],
    output_pod5: Union[str, Path] = "FAST5s_to_POD5.pod5"
) -> None:
    """
    Convert Nanopore FAST5 files (single file, list of files, or directory)
    into a single .pod5 output using the 'pod5 convert fast5' CLI tool.
    """

    output_pod5 = str(output_pod5)  # ensure string

    # 1) If user gives a list of FAST5 files
    if isinstance(fast5_dir, (list, tuple)):
        fast5_paths = [str(Path(f)) for f in fast5_dir]
        cmd = ["pod5", "convert", "fast5", *fast5_paths, "--output", output_pod5]
        subprocess.run(cmd, check=True)
        return

    # Ensure Path object
    p = Path(fast5_dir)

    # 2) If user gives a single file
    if p.is_file():
        cmd = ["pod5", "convert", "fast5", str(p), "--output", output_pod5]
        subprocess.run(cmd, check=True)
        return

    # 3) If user gives a directory → collect FAST5s
    if p.is_dir():
        fast5_paths = sorted(str(f) for f in p.glob("*.fast5"))
        if not fast5_paths:
            raise FileNotFoundError(f"No FAST5 files found in {p}")

        cmd = ["pod5", "convert", "fast5", *fast5_paths, "--output", output_pod5]
        subprocess.run(cmd, check=True)
        return

    raise FileNotFoundError(f"Input path invalid: {fast5_dir}")

