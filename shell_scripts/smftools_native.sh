###################################################################################################
#### User defined parameters ####

## Input files ##
# Path to directory containing input POD5 files
BASE_dir='/Users/base_dir_for_analysis'
# Path to directory containing input POD5 files
POD5_dir="${BASE_dir}/pod5"
# Path to reference FASTA
FASTA='/Users/path_to_FASTA' 

## Output files ##
# Path to your output directory for the overall analysis
output_directory="${BASE_dir}dir_name" 

## dorado basecaller parameters ##
# Path to your basecalling model
model='/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0' 
# Barcoding kit
barcode_kit='SQK-NBD114-24'

## Path to accompanying python scripts ##
script_dir='/smftools/python_scripts'
sep_BAM_by_BC="${script_dir}/separate_BAM_by_tag.py"
FORMAT_SMF="${script_dir}/Modkit_extract_to_anndata.py"

## Thresholds for canonical and modified basecalls in Modkit (Ranges from 0-1 as a call probability) ##
filter_threshold=0.8
m6A_threshold=0.8
5mC_threshold=0.8
5hmC_threshold=0.8

## Final anndata object parameters ##
# An experiment name for the final h5ad file
EXPERIMENT_NAME='SMF_pilot' 
# Modifications to analyze
MOD_LIST=("6mA" "5mC_5hmC")
# The number of sample TSV files to load into memory and ouput as an anndata at a time
BATCH_SIZE=4 
# The lower modification call probability to act as a passing threshold
MOD_THRESHOLD=0.8 
#################################

#################################
#### Other variables ####
mod_BAM="${output_directory}/HAC_mod_calls"
aligned_mod_BAM="${mod_BAM}_aligned"
aligned_sorted_mod_BAM="${aligned_mod_BAM}_sorted"
split_dir="${output_directory}/split_BAMS"
mod_bed_dir="${output_directory}/split_mod_beds"
mod_tsv_dir="${output_directory}/split_mod_tsvs"
#################################
###################################################################################################

###################################################################################################
### Exectution ###
cd $BASE_dir
if [ ! -d "$output_directory" ]; then
    mkdir "$output_directory"
    echo "Directory '$output_directory' created successfully."
else
    echo "Directory '$output_directory' already exists."
fi

BAM_SUFFIX='.bam'

### 1) Dorado modified base calling ###
# Outputs a BAM file of modified reads (Not a standard BAM style alignment file)
# $model variable will be the path to the general model
# --modified-bases option will precede a space separated list of all modified bases that we want.
# --kit-name option detects barcodes and adds it to the supplement bc tag in the output BAM.
# -Y option enables soft clipping for minimap2.
dorado basecaller $model $POD5_dir --kit-name $barcode_kit -Y --modified-bases ${MOD_LIST[@]} > ''$mod_BAM''$BAM_SUFFIX''

### 2) Aligning a modified BAM file to a reference using Dorado ###
# Takes an input modified BAM file output from Dorado, as well as a minimap2 index for the reference FASTA.
# Outputs a position sorted BAM and a .bai file.
# Also makes a bed file for the aligned reads.
# Align the modified BAM to a reference FASTA and output an aligned BAM
dorado aligner --secondary=no $FASTA ''$mod_BAM''$BAM_SUFFIX'' > ''$aligned_mod_BAM''$BAM_SUFFIX''
# Sort the BAM on positional coordinates
samtools sort -o ''$aligned_sorted_mod_BAM''$BAM_SUFFIX'' ''$aligned_mod_BAM''$BAM_SUFFIX''
# Create a BAM index file
samtools index ''$aligned_sorted_mod_BAM''$BAM_SUFFIX''
# Make a bed file of coordinates for the BAM
samtools view ''$aligned_sorted_mod_BAM''$BAM_SUFFIX'' | awk '{print $3, $4, $4+length($10)-1}' > "${aligned_sorted_mod_BAM}_bed.bed"
# Make a text file of reads for the BAM
samtools view ''$aligned_sorted_mod_BAM''$BAM_SUFFIX'' | cut -f1 | > "${aligned_sorted_mod_BAM}_read_names.txt"

### 3) Splitting modified BAM files by barcode ###
# Takes an aligned, sorted, modified BAM file as input.
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
python3 $sep_BAM_by_BC ''$aligned_sorted_mod_BAM''$BAM_SUFFIX'' $aligned_sorted_mod_BAM
# Make a BAM index file for the BAMs in that directory
find "$split_dir" -type f -name "*$BAM_SUFFIX" | while IFS= read -r input_file; do
    samtools index $input_file
done

### 4) Using nanopore modkit to work with modified BAM files ###
# Takes an aligned, sorted, modified BAM file and outputs a variety of tables and statistics
# Output the percentile of bases falling at a call threshold (threshold is a probability between 0-1) for the overall BAM file. It is generally good to look at these parameters on positive and negative controls.
modkit sample-probs ''$aligned_sorted_mod_BAM''$BAM_SUFFIX''
modkit summary ''$aligned_sorted_mod_BAM''$BAM_SUFFIX'' --filter-threshold $filter_threshold --mod-thresholds "m:$m5C_threshold" --mod-thresholds "a:$m6A_threshold" --mod-thresholds "h:$hm5C_threshold"
# Make an output directory for the split modBeds.
cd $output_directory
if [ ! -d "$mod_bed_dir" ]; then
    mkdir "$mod_bed_dir"
    echo "Directory '$mod_bed_dir' created successfully."
else
    echo "Directory '$mod_bed_dir' already exists."
fi
# Generating Barcode summaries starting from the overall BAM file that was direct output of dorado aligner#
modkit pileup ''$aligned_sorted_mod_BAM''$BAM_SUFFIX'' $mod_bed_dir --partition-tag BC --only-tabs --filter-threshold $filter_threshold --mod-thresholds "m:$m5C_threshold" --mod-thresholds "a:$m6A_threshold" --mod-thresholds "h:$hm5C_threshold"
# Make an output directory for the split Modkit extract TSVs.
cd $output_directory
if [ ! -d "$mod_tsv_dir" ]; then
    mkdir "$mod_tsv_dir"
    echo "Directory '$mod_tsv_dir' created successfully."
else
    echo "Directory '$mod_tsv_dir' already exists."
fi
# Extract methylations calls for split BAM files
cd $mod_tsv_dir
find "$split_dir" -type f -name "*$BAM_SUFFIX" | while IFS= read -r input_file; do
    echo $input_file
    file_name=$(basename "$input_file")
    output_tsv_temp=''$mod_tsv_dir'/'$file_name''
    output_tsv="${output_tsv_temp//"$BAM_SUFFIX"/}_extract.tsv"
    # modkit summary gives the overall base call features in the overall input BAM file.
    modkit summary $input_file
    # modkit extract gives read level modified base information for every base in an input BAM file. These output TSVs are about 10X larger than the input BAM
    modkit extract --filter-threshold $filter_threshold --mod-thresholds "m:$m5C_threshold" --mod-thresholds "a:$m6A_threshold" --mod-thresholds "h:$hm5C_threshold" $input_file null --read-calls $output_tsv
    # Zip the output TSV 
    echo 'zipping '$output_tsv''
    zip "$output_tsv.zip" $output_tsv
    # Delete the non-zipped TSV
    echo 'removing '$output_tsv''
    rm $output_tsv
done

### 5) Binarizing single molecule modified base data across conditions and writing out anndata object ###
cd $mod_tsv_dir
python3 $FORMAT_SMF $MOD_LIST $FASTA $BATCH_SIZE $MOD_THRESHOLD $EXPERIMENT_NAME