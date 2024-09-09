# subsample_pod5

def subsample_pod5(pod5_path, read_name_path, output_directory):
    """
    Takes a POD5 file and a text file containing read names of interest and writes out a subsampled POD5 for just those reads.
    This is a useful function when you have a list of read names that mapped to a region of interest that you want to reanalyze from the pod5 level.

    Parameters:
        pod5_path (str): File path to the POD5 to subsample.
        read_name_path (str | int): File path to a text file of read names. One read name per line. If an int value is passed, a random subset of that many reads will occur
        output_directory (str): A file path to the directory to output the file.

    Returns:
        None
    """
    import pod5 as p5
    import os

    input_pod5_base = os.path.basename(pod5_path)

    if type(read_name_path) == str:
        input_read_name_base = os.path.basename(read_name_path)
        output_base = input_pod5_base.split('.pod5')[0] + '_' + input_read_name_base.split('.txt')[0] + '_subsampled.pod5'
        # extract read names into a list of strings
        with open(read_name_path, 'r') as file:
            read_names = [line.strip() for line in file]
        with p5.Reader(pod5_path) as reader:
            read_records = []
            for read_record in reader.reads(selection=read_names):
                read_records.append(read_record.to_read())

    elif type(read_name_path) == int:
        import random
        output_base = input_pod5_base.split('.pod5')[0] + f'_{read_name_path}_randomly_subsampled.pod5'
        with p5.Reader(pod5_path) as reader:
            all_read_records = []
            for read_record in reader.reads():
                all_read_records.append(read_record.to_read())
        if read_name_path <= len(all_read_records):
            read_records = random.sample(all_read_records, read_name_path)
        else:
            print('Trying to sample more reads than are contained in the input pod5, please try a lower value.')

    output_pod5 = os.path.join(output_directory, output_base)

    # Write the subsampled POD5
    with p5.Writer(output_pod5) as writer:
        writer.add_reads(read_records)