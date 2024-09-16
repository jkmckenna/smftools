# fast5_to_pod5

def fast5_to_pod5(fast5_dir, output_pod5='FAST5s_to_POD5.pod5'):
    """
    Convert Nanopore FAST5 files to POD5 file

    Parameters:
        fast5_dir (str): String representing the file path to a directory containing all FAST5 files to convert into a single POD5 output.
        output_pod5 (str): The name of the output POD5.
    
    Returns:
        None
    
    """
    import subprocess
    from pathlib import Path

    if Path(fast5_dir).is_file():
        subprocess.run(["pod5", "convert", "fast5", fast5_dir, "--output", output_pod5])
    elif Path(fast5_dir).is_dir():
        subprocess.run(["pod5", "convert", "fast5", f".{fast5_dir}*.fast5", "--output", output_pod5])
