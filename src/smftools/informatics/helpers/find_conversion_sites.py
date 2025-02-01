def find_conversion_sites(fasta_file, modification_type, conversion_types):
    """
    Efficiently finds genomic coordinates of cytosines (5mC) or adenines (6mA) in a reference FASTA file.

    Parameters:
        fasta_file (str): Path to the converted reference FASTA.
        modification_type (str): Modification type of interest ('5mC' or '6mA').
        conversion_types (list): List of conversion types. The first element represents the unconverted record name.

    Returns: 
        dict: Dictionary where keys are unconverted record IDs and values are lists containing:
              [sequence length, top strand coordinates, bottom strand coordinates, sequence, complement sequence].
    """
    import numpy as np
    from Bio import SeqIO

    unconverted = conversion_types[0]
    record_dict = {}

    # Define base mapping based on modification type
    base_mappings = {
        '5mC': ('C', 'G'),  # Cytosine and Guanine
        '6mA': ('A', 'T')   # Adenine and Thymine
    }

    if modification_type not in base_mappings:
        raise ValueError("Invalid modification_type. Choose '5mC' or '6mA'.")

    top_base, bottom_base = base_mappings[modification_type]

    # Read FASTA file and process records
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            if unconverted in record.id:
                sequence = str(record.seq).upper()
                complement = str(record.seq.complement()).upper()
                sequence_length = len(sequence)

                # Convert sequence to NumPy array for fast indexing
                seq_array = np.array(list(sequence))
                top_strand_coordinates = np.where(seq_array == top_base)[0].tolist()
                bottom_strand_coordinates = np.where(seq_array == bottom_base)[0].tolist()

                # Store results
                record_dict[record.id] = [sequence_length, top_strand_coordinates, bottom_strand_coordinates, sequence, complement]

    return record_dict
