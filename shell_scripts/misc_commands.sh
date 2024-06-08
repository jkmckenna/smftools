### JM Compilation of useful nanopore commands ###
### Extra functions of potential use ##
# A) Basic concatenation of split fastqs in a directory. Keeps only reads of a given length and quality.
# B) Trims out adapter and barcode sequences from a fastq file.
# C) Extra end trimming step, optional.
# D) Removing all files of a given type in a directory and all subdirectories.
# E) Reformatting a FASTA to have the proper text wrapping
# F) Creating a FASTA index file
# G) Parse a BAM file to extract information of interest
# H) Removing a list of read names from a FASTQ
# I) Concatenating and deleting h5ad files
###################################################################################################

###################################################################################################
### A) Concatenate FASTQSs in input_directory and output into a subdirectory named concatenated###
# -x option recursively searches all subdirectories for FASTQs (remove this option if doing a run without barcoding)
# -d option creates a new directory to contain the concatenated FASTQs for each barcode. In the case below, the subdirectory will be named 'concatenated'
# -f option creates a txt file with basic metrics from each file in the analysis. In the case below, the file will be named 'file_summaries'
# -a option is the minimum read length that will be included in the concatenated FASTQ output
# -q option is the minimum Q-score that will be included in the concatenated FASTQ output
# Final file path is the directory in which to recursively search subdirectories for FASTQs to concatenate
input_directory="/path_to_FASTQs"
cd "$input_directory"
fastcat -x -d concatenated -f file_summaries -a 200 -q 8 "$input_directory" 
###################################################################################################
###################################################################################################
### B) Trim out adapter and barcode sequences from concatenated fastqs ###
# finds all files nested in the input_directory that end in the matched file_type suffix. 
# For each of these files, make an output file in the same directory that has _porechop appended to the suffix before the file_type string
# -i preceeds the input file
# -o preceeds the output file
# --extra_end_trim option specifies how many extra bases to remove from the read after the barcode or adapter.
input_directory="/Path_to_FASTQs"
file_type=".fastq.gz"
find "$input_directory" -type f -name "*$file_type" | while IFS= read -r input_file; do
    porechop -i "$input_file" -o  "${input_file//"$file_type"/}_porechop$file_type"
done
###################################################################################################
###################################################################################################
### C) Extra end trimming of reads ###
# Use seqkit
# -d prefaces the 1-indexed coordinates to delete
# -X indicates the input file path
# -o indicates the output file path
input_directory="/Path_to_FASTQs"
file_type=".fastq.gz"
find "$input_directory" -type f -name "*$file_type" | while IFS= read -r input_file; do
    seqkit stats -a $input_file -T | csvtk csv2md -t
done
find "$input_directory" -type f -name "*porechop$file_type" | while IFS= read -r input_file; do
    cutadapt -u 20 -u -20 -o "${input_file//"$file_type"/}_trimmed$file_type" $input_file
done
###################################################################################################
###################################################################################################
### D) Removing all files of a given type in a directory and all subdirectories ###
input_directory='/Path_to_input_directory'
file_type="filtered"
find "$input_directory" -type f -name "*$file_type*" | while IFS= read -r input_file; do
    rm $input_file
done
###################################################################################################
###################################################################################################
### E) Taking a FASTA and ensuring that it is in the correct format before index formation.
FASTA_PATH='FASTA_path'
seqkit seq -w 60 -u $FASTA_PATH -o "${FASTA_PATH//".fa"/}_reformatted.fa"
###################################################################################################
###################################################################################################
### F) Indexing a FASTA file and outputing the index into the same directory ###
FASTA_PATH='/Path_to_FASTA'
samtools faidx $FASTA_PATH -o "$FASTA_PATH.fai"
###################################################################################################
###################################################################################################
### G) Parsing a BAM file ###
input_directory="/Path_to_input_directory"
BAM_PATH='/Path_to_BAM'
# Accompanying python script
modified_BAM_parse='/Modified_BAM_parse.py' 
python3 $modified_BAM_parse $BAM_PATH
###################################################################################################
###################################################################################################
### H) Removing a list of reads from a FASTQ ###
conda activate genomics
input_directory='/Path_to_FASTQs'
# Path to text file with read names to remove from a FASTQ
REMOVE_READ_PATH='/Path_to_text_file' 
# Accompanying python script
remove_reads_from_FASTQ='/remove_reads_from_FASTQ.py' 
file_type=".fastq.gz"
find "$input_directory" -type f -name "*$file_type*" | while IFS= read -r input_file; do
    python3 $remove_reads_from_FASTQ $input_file $REMOVE_READ_PATH "${input_file//"$file_type"/}_reads_removed.fastq.gz"
done
###################################################################################################
###################################################################################################
### I) Concatenating and deleting h5ad files ###
input_directory='/Path_to_h5ad_dir'
file_type=".fastq.gz"
delete_input_files="False"
# Accompanying python script
concatenate_h5ads='/Concatenate_h5ads.py' 

cd $input_directory
python3 $concatenate_h5ads $file_type $delete_input_files
###################################################################################################
######################################################################################################################################################################################################
