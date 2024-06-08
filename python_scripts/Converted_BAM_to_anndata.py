######################################################################################################
from Bio import SeqIO
import pandas as pd
import pysam
import numpy as np
import anndata as ad
from datetime import datetime
import os
import argparse
######################################################################################################
# Get the current date
current_date = datetime.now()
# Format the date as a string
date_string = current_date.strftime("%Y%m%d")
date_string = date_string[2:]
######################################################################################################

######################################################################################################
def time_string():
    current_time = datetime.now()
    return current_time.strftime("%H:%M:%S")

def count_aligned_reads(bam_file):
    """
    Counts the number of aligned/unaligned reads in a BAM file. Also returns a dictionary, keyed by reference id that points to a tuple. The tuple contains an integer number of mapped reads to that reference, followed by the proportion of mapped reads that map to that reference
    """
    print('{0}: Counting aligned reads in BAM > {1}'.format(time_string(), bam_file))
    aligned_reads_count = 0
    unaligned_reads_count = 0
    # Make a dictionary, keyed by the reference_name of reference chromosome that points to an integer number of read counts mapped to the chromosome
    record_counts = {}
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        # Iterate over reads to get the total mapped read counts and the reads that map to each reference
        for read in bam:
            if read.is_unmapped: 
                unaligned_reads_count += 1
            else: 
                aligned_reads_count += 1
                if read.reference_name in record_counts:
                    record_counts[read.reference_name] += 1
                else:
                    record_counts[read.reference_name] = 1
        # reformat the dictionary to contain read counts mapped to the reference, as well as the proportion of mapped reads in reference
        for reference in record_counts:
            proportion_mapped_reads_in_record = record_counts[reference] / aligned_reads_count
            record_counts[reference] = (record_counts[reference], proportion_mapped_reads_in_record)
    return aligned_reads_count, unaligned_reads_count, record_counts

def find_coordinates(fasta_file, modification_type):
    """
    A function to find genomic coordinates in every unconverted record contained within a FASTA file of C in the GpC and not GpCpG context.
    If searching for adenine conversions, it will find coordinates of all adenines.
    Returns: 
    A dictionary called record_dict, which is keyed by unconverted record ids contained within the FASTA. Points to a list containing: 1) sequence length of the record, 2) top strand coordinate list, 3) bottom strand coorinate list
    """
    print('{0}: Finding positions of interest in reference FASTA > {1}'.format(time_string(), fasta_file))
    # Initialize lists to hold top and bottom strand positional coordinates of interest
    top_strand_coordinates = []
    bottom_strand_coordinates = []
    record_dict = {}
    print('{0}: Opening FASTA file {1}'.format(time_string(), fasta_file))
    # Open the FASTA record as read only
    with open(fasta_file, "r") as f:
        # Iterate over records in the FASTA
        for record in SeqIO.parse(f, "fasta"):
            # Only iterate over the unconverted records for the reference
            if 'unconverted' in record.id:
                print('{0}: Iterating over record {1} in FASTA file {2}'.format(time_string(), record, fasta_file))
                # Extract the sequence string of the record
                sequence = str(record.seq).upper()
                sequence_length = len(sequence)
                if modification_type == '5mC':
                    # Iterate over the sequence string from the record
                    for i in range(1, len(sequence)-1):
                        if sequence[i] == 'C' and sequence[i-1] == 'G' and (i == len(sequence) - 1 or sequence[i+1] != 'G'):
                            top_strand_coordinates.append(i)  # 0-indexed coordinate
                            
                        if sequence[i] == 'G' and sequence[i+1] == 'C' and sequence[i-1] != 'C':
                            bottom_strand_coordinates.append(i)  # 0-indexed coordinate      
                    print('{0}: Returning zero-indexed top and bottom strand FASTA coordinates for cytosines of interest'.format(time_string()))
                elif modification_type == '6mA':
                    # Iterate over the sequence string from the record
                    for i in range(0, len(sequence)):
                        if sequence[i] == 'A':
                            top_strand_coordinates.append(i)  # 0-indexed coordinate
                        if sequence[i] == 'T':
                            bottom_strand_coordinates.append(i)  # 0-indexed coordinate      
                    print('{0}: Returning zero-indexed top and bottom strand FASTA coordinates for adenines of interest'.format(time_string()))          
                else:
                    print('modification_type not found. Please try 5mC or 6mA')    
                record_dict[record.id] = [sequence_length, top_strand_coordinates, bottom_strand_coordinates]
            else:
                pass  
    return record_dict
            
def extract_base_identity_at_coordinates(bam_file, chromosome, positions, max_reference_length):
    """
    Takes an input position sorted BAM file, chromosome number, position coordinate set, and reference length to extract the base identitity from the read.
    Outputs a dictionary, keyed by positional identity, that points to a list of Base identities from each read.
    If the read does not contain that position, fill the list at that index with a np.nan value.
    """
    positions = set(positions)
    # Initialize a base identity dictionary that will hold key-value pairs that are: key (read-name) and value (list of base identities at positions of interest)
    base_identities = {}
    # Open the postion sorted BAM file
    print('{0}: Reading BAM file: {1}'.format(time_string(), bam_file))
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        # Iterate over every read in the bam that comes from the chromosome of interest
        print('{0}: Iterating over reads in bam'.format(time_string()))
        for read in bam.fetch(chromosome): 
            if read.query_name in base_identities:
                print('Duplicate read found in BAM for read {}. Skipping duplicate'.format(read.query_name))
            else:          
                # Initialize the read key in the base_identities dictionary by pointing to a nan filled list of length reference_length
                base_identities[read.query_name] = [np.nan] * max_reference_length
                # Iterate over a list of tuples for the given read. The tuples contain the 0-indexed position relative to the read start, as well the 0-based index relative to the reference.
                for read_position, reference_position in read.get_aligned_pairs():
                    # If the aligned read's reference coordinate is in the positions set and if the read position was successfully mapped
                    if reference_position in positions and read_position:
                        # get the base_identity in the read corresponding to that position
                        base_identity = read.query_sequence[read_position]
                        # Add the base identity to array
                        base_identities[read.query_name][reference_position] = base_identity
    return base_identities

def binarize_base_identities(base_identities, strand, modification_type):
    """
    Takes the base identities dictionary and returns a binarized format of the dictionary
    """
    binarized_base_identities = {}
    # Iterate over base identity keys to binarize the base identities
    for key in base_identities.keys():
        if strand == 'top':
            if modification_type == '5mC':
                binarized_base_identities[key] = [1 if x == 'C' else 0 if x == 'T' else np.nan for x in base_identities[key]]
            elif modification_type == '6mA':
                binarized_base_identities[key] = [1 if x == 'A' else 0 if x == 'G' else np.nan for x in base_identities[key]]
        elif strand == 'bottom':
            if modification_type == '5mC':
                binarized_base_identities[key] = [1 if x == 'G' else 0 if x == 'A' else np.nan for x in base_identities[key]]
            elif modification_type == '6mA':
                binarized_base_identities[key] = [1 if x == 'T' else 0 if x == 'C' else np.nan for x in base_identities[key]]
        else:
            pass
    return binarized_base_identities
######################################################################################################

######################################################################################################
if __name__ == "__main__":
    ### Parse Inputs ###
    parser = argparse.ArgumentParser(description="Convert BAM files to binarized anndata")
    parser.add_argument("input_fasta", help="converted FASTA file")
    parser.add_argument("input_bams", help="Directory containing input BAMs to binarize")
    parser.add_argument("mapping_threshold", help="Minimal threshold of mapped reads to a reference chromosome to allow")
    parser.add_argument("experiment_name", help="String to append to the output h5ad file")
    args = parser.parse_args()
    converted_fasta_file = args.input_fasta
    bam_directory = args.input_bams
    mapping_threshold = args.mapping_threshold
    experiment_name = args.experiment_name
    bam_suffix = '.bam'
# Get all of the input BAM files
files = os.listdir(bam_directory)
# Change directory to the BAM directory
os.chdir(bam_directory)
# Filter file names that contain the search string in their filename and keep them in a list
bams = [bam for bam in files if bam_suffix in bam and '.bai' not in bam]
# Sort file list by names and print the list of file names
bams.sort()
print(bams)
modification_types = ['5mC', '6mA'] # Options include 6mA and 5mC
strands = ['top', 'bottom'] # Options include top and bottom
final_adata = None
# Make a dictionary, keyed by modification type, that points to another dictionary of unconverted_record_ids. This points to a list of: 1) record length, 2) top strand conversion coordinates, 3) bottom strand conversion coordinates
modification_dict = {}
# While populating the dictionary, also extract the longest sequence record in the input references
max_reference_length = 0
for modification_type in modification_types:
    modification_dict[modification_type] = find_coordinates(converted_fasta_file, modification_type)
    for record in modification_dict[modification_type].keys():
        if modification_dict[modification_type][record][0] > max_reference_length:
            max_reference_length = modification_dict[modification_type][record][0]
######################################################################################################

######################################################################################################
# Iterate over the experiment BAM files
for bam_index, bam in enumerate(bams):
    # Give each bam a sample name
    sample = bam.split(sep=bam_suffix)[0]
    # look at aligned read proportions in the bam
    aligned_reads_count, unaligned_reads_count, record_counts = count_aligned_reads(bam)
    print('{} percent of reads in bam aligned successfully'.format(aligned_reads_count*100 / (aligned_reads_count+unaligned_reads_count)))
    records_to_analyze = []
    # Iterate over converted reference strands and decide which to use in the analysis based on the mapping_threshold
    for record in record_counts:
        print('{0} reads mapped to reference record {1}. This is {2} percent of all mapped reads'.format(record_counts[record][0], record, record_counts[record][1]*100))
        if record_counts[record][1] >= mapping_threshold:
            records_to_analyze.append(record)
    # Iterate over records to analyze (ie all conversions detected)
    for record in records_to_analyze:
        mod_type, strand = record.split('_')[-2:]
        if strand == 'top':
            strand_index = 1
        elif strand == 'bottom':
            strand_index = 2
        chromosome = record.split('_{0}_{1}'.format(mod_type, strand))[0]
        unconverted_chromosome_name = chromosome + '_unconverted_top'
        positions = modification_dict[mod_type][unconverted_chromosome_name][strand_index]
        # Get a dictionary of positional identities keyed by read id
        base_identities = extract_base_identity_at_coordinates(bam, record, positions, max_reference_length) 
        # binarize the dictionary of positional identities
        binarized_base_identities = binarize_base_identities(base_identities, strand, mod_type) 
        # converts the base identity dictionary to a dataframe.
        binarized_base_identities_df = pd.DataFrame.from_dict(binarized_base_identities, orient='index') 
        # Load an anndata object with the sample data
        X = binarized_base_identities_df.values
        adata = ad.AnnData(X, dtype=X.dtype)
        adata.obs_names = binarized_base_identities_df.index
        adata.obs_names = adata.obs_names.astype(str)
        adata.var_names = binarized_base_identities_df.columns
        adata.var_names = adata.var_names.astype(str)
        adata.obs['Sample'] = [sample] * len(adata)
        adata.obs['Strand'] = [strand] * len(adata)
        adata.obs['Dataset'] = [mod_type] * len(adata)
        adata.obs['Reference'] = [record] * len(adata)
        if final_adata:
            final_adata = ad.concat([final_adata, adata], join='outer', index_unique=None)
        else:
            final_adata = adata
######################################################################################################

######################################################################################################

## Export the final adata object
final_adata.write_h5ad('{0}_{1}.h5ad.gz'.format(date_string, experiment_name), compression='gzip')