# fast5_to_pod5

def fast5_to_pod5(fast5_dir, output_dir='outputs/', output_pod5='FAST5s_to_POD5.pod5'):
    """
    Convert Nanopore FAST5 files to POD5 file

    Parameters:
        fast5_dir (str): String representing the file path to a directory containing all FAST5 files to convert into a single POD5 output.
        output_dir (str): String representing the file path to the output directory.
        output_pod5 (str): The name of the output POD5 to write out within the output directory.
    
    Returns:
        None
    
    """
    import subprocess
    import os
    pod5 = os.path.join(output_dir, output_pod5)
    subprocess.run(["pod5", "convert", "fast5", f".{fast5_dir}*.fast5", "--output", pod5])
