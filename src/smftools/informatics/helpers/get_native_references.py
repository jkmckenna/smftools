## get_native_references

# Direct methylation specific
def get_native_references(fasta_file):
    """
    Makes a dictionary keyed by record id which points to the record length and record sequence.

    Paramaters:
        fasta_file (str): A string representing the path to the FASTA file for the experiment.

    Returns: 
        None
    """
    from .. import readwrite
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    record_dict = {}
    print('{0}: Opening FASTA file {1}'.format(readwrite.time_string(), fasta_file))
    # Open the FASTA record as read only
    with open(fasta_file, "r") as f:
        # Iterate over records in the FASTA
        for record in SeqIO.parse(f, "fasta"):
            # Extract the sequence string of the record
            sequence = str(record.seq).upper()
            sequence_length = len(sequence) 
            record_dict[record.id] = [sequence_length, sequence]
    return record_dict