######################################################################################################################################################################################################
### Workflow for Nanopore Standard base calling SMF ###
###################################################################################################
#### User defined parameters ####

BASE_dir='/Users/base_dir_for_analysis'
# Path to directory containing input POD5 files
POD5_dir="${BASE_dir}/pod5"
# Path to reference FASTA
FASTA="${BASE_dir}/.fa"

## Output files ##
# Path to your output directory for the overall analysis
output_directory="${BASE_dir}dir_name" 
# Path to the anticipated output converted FASTA file
converted_FASTA='' 

## Conversion parameters ##
# Options include 6mA, 5mC, unconverted
modification_types="unconverted,5mC,6mA"
# Options include top and bottom
strands="top,bottom"

## dorado basecaller parameters ##
# Path to your basecalling model
model='/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0' 
# Barcoding kit
barcode_kit='SQK-NBD114-24'

## Path to accompanying python scripts ##
script_dir='/smftools/python_scripts'
convert_FASTA="${script_dir}/Generate_converted_FASTA.py"
BAM_to_anndata="${script_dir}/Converted_BAM_to_anndata.py" 
sep_BAM_by_BC="${script_dir}/separate_BAM_by_tag.py" 

## Minimum proportion of reads mapping to a reference to further use that reference (Ranges from 0-1 as a proportion of mapped reads) ##
mapping_threshold=0.05

## Final anndata object parameters ##
# An experiment name for the final h5ad file
EXPERIMENT_NAME='SMF_pilot' 

#################################
#### Other variables ####
BAM="${output_directory}/HAC_basecalls"
aligned_BAM="${BAM}_aligned"
aligned_sorted_BAM="${aligned_BAM}_sorted"
split_dir="${output_directory}/split_BAMS" 
#################################

###################################################################################################
### Execution ###
BAM_SUFFIX='.bam'
### 1) Generate a conferted FASTA from a reference FASTA ###
python3 $convert_FASTA $FASTA $modification_types $strands $converted_FASTA

### 2) Basecall from the input POD5 to generate a singular output BAM ###
dorado basecaller $model $POD5_dir --kit-name $barcode_kit > ''$BAM''$BAM_SUFFIX''

### 3) Align the BAM to the converted reference FASTA ###
# Takes an input BAM file output from Dorado, as well as a converted reference FASTA.
# Outputs a position sorted BAM and a .bai file.
# Also makes a bed file for the aligned reads.
# Align the modified BAM to a reference FASTA and output an aligned BAM
dorado aligner --secondary=no $converted_FASTA ''$BAM''$BAM_SUFFIX'' > ''$aligned_BAM''$BAM_SUFFIX''
# Sort the BAM on positional coordinates
samtools sort -o ''$aligned_sorted_BAM''$BAM_SUFFIX'' ''$aligned_BAM''$BAM_SUFFIX''
# Create a BAM index file
samtools index ''$aligned_sorted_BAM''$BAM_SUFFIX''
# Make a bed file of coordinates for the BAM
samtools view ''$aligned_sorted_BAM''$BAM_SUFFIX'' | awk '{print $3, $4, $4+length($10)-1}' > "${aligned_sorted_BAM}_bed.bed"
# Make a text file of reads for the BAM
samtools view ''$aligned_sorted_BAM''$BAM_SUFFIX'' | cut -f1 | > "${aligned_sorted_BAM}_read_names.txt"

### 4) Split BAM files by barcode ###
# Takes an aligned, sorted, BAM file as input.
# Outputs a directory of BAM files split on barcode value in the BC tag
# Make an output directory for the split BAMs.
cd $output_directory
if [ ! -d "$split_dir" ]; then
    mkdir "$split_dir"
    echo "Directory '$split_dir' created successfully."
else
    echo "Directory '$split_dir' already exists."
fi
# Change directory into the output directory and split the BAMs into that directory
cd $split_dir
python3 $sep_BAM_by_BC ''$aligned_sorted_BAM''$BAM_SUFFIX'' $aligned_sorted_BAM
# Make a BAM index file for the BAMs in that directory
find "$split_dir" -type f -name "*$BAM_SUFFIX" | while IFS= read -r input_file; do
    samtools index $input_file
done

### 5) Take the directory with all of the BAM files and create a binarized anndata object for the experiment ###
cd $split_dir
python3 $BAM_to_anndata $converted_FASTA $split_dir $mapping_threshold $EXPERIMENT_NAME
###################################################################################################