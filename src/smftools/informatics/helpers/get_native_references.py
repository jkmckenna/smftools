## get_native_references
from .. import readwrite
# bioinformatic operations
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Direct methylation specific
def get_native_references(fasta_file):
    """
    Input: A FASTA file
    Returns: 
    A dictionary called record_dict, which is keyed by record ids contained within the FASTA. Points to a list containing: 1) sequence length of the record, 2) sequence of the record
    """
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