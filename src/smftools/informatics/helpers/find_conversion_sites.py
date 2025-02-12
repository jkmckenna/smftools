def find_conversion_sites(fasta_file, modification_type, conversion_types):
    """
    Finds genomic coordinates of modified bases (5mC or 6mA) in a reference FASTA file.

    Parameters:
        fasta_file (str): Path to the converted reference FASTA.
        modification_type (str): Modification type ('5mC' or '6mA') or 'unconverted'.
        conversion_types (list): List of conversion types. The first element is the unconverted record type.

    Returns: 
        dict: Dictionary where keys are **both unconverted & converted record names**.
              Values contain:
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

    # Read FASTA file and process records
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            if unconverted in record.id:
                sequence = str(record.seq).upper()
                complement = str(record.seq.complement()).upper()
                sequence_length = len(sequence)

                # Unconverted case: store the full sequence without coordinate filtering
                if modification_type == unconverted:
                    record_dict[record.id] = [sequence_length, [], [], sequence, complement]

                # Process converted records: extract modified base positions
                elif modification_type in base_mappings:
                    top_base, bottom_base = base_mappings[modification_type]
                    seq_array = np.array(list(sequence))
                    top_strand_coordinates = np.where(seq_array == top_base)[0].tolist()
                    bottom_strand_coordinates = np.where(seq_array == bottom_base)[0].tolist()

                    record_dict[record.id] = [sequence_length, top_strand_coordinates, bottom_strand_coordinates, sequence, complement]

                else:
                    raise ValueError(f"Invalid modification_type: {modification_type}. Choose '5mC', '6mA', or 'unconverted'.")

    return record_dict
