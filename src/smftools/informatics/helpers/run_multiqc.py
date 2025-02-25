def run_multiqc(input_dir, output_dir):
    """
    Runs MultiQC on a given directory and saves the report to the specified output directory.

    Parameters:
    - input_dir (str): Path to the directory containing QC reports (e.g., FastQC, Samtools, bcftools outputs).
    - output_dir (str): Path to the directory where MultiQC reports should be saved.

    Returns:
    - None: The function executes MultiQC and prints the status.
    """
    import os
    import subprocess
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct MultiQC command
    command = ["multiqc", input_dir, "-o", output_dir]

    print(f"Running MultiQC on '{input_dir}' and saving results to '{output_dir}'...")
    
    # Run MultiQC
    try:
        subprocess.run(command, check=True)
        print(f"MultiQC report generated successfully in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running MultiQC: {e}")

