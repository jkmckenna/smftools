import pandas as pd
import argparse
from io import StringIO

columns = ['chrom', 'start position', 'end position', 'modified base code and motif', 'score', 'strand', 'start position dup', 'end position dup', 'color', 'Nvalid_cov', 'fraction modified', 'Nmod', 'Ncanonical', 'Nother_mod', 'Ndelete', 'Nfail', 'Ndiff', 'Nnocall']

def Filter_modBed(input_bed, filter_column, filter_threshold):
    # Read the input BED file into a pandas DataFrame
    bed_df = pd.read_csv(input_bed, sep='\t', header=None, names=columns)
    # Filter rows based on the threshold value in the "score" column
    filtered_bed_df = bed_df[bed_df[filter_column] >= int(filter_threshold)]
    output_bed = input_bed_name.split('.bed')[0] + '_filtered.bed'
    # Write the filtered DataFrame to a new BED file
    filtered_bed_df.to_csv(output_bed, sep="\t", header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a new Bed file filtered on certain column thresholds")
    parser.add_argument("input_bed", help="Path to the input modBed file to filter.")
    parser.add_argument("filter_column", help="Column to filter by.")
    parser.add_argument("filter_threshold", help="Threshold to filter by.")
    args = parser.parse_args()
    with open(args.input_bed, "r") as file:
        bed = file.read()
    bed_with_tabs = bed.replace(" ", "\t")
    input_bed = StringIO(bed_with_tabs)
    input_bed_name = args.input_bed
    Filter_modBed(input_bed, args.filter_column, args.filter_threshold)