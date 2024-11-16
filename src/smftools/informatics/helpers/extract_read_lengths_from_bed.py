# extract_read_lengths_from_bed

def extract_read_lengths_from_bed(file_path):
    """
    Load a dict of read names that points to the read length

    Params:
        file_path (str): file path to a bed file
    Returns:
        read_dict (dict)
    """
    import pandas as pd
    columns = ['chrom', 'start', 'end', 'length', 'name']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns, comment='#')
    read_dict = {}
    for _, row in df.iterrows():
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        name = row['name']
        length = row['length']
        read_dict[name] = length

    return read_dict
        
