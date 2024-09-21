# subsample_pod5

def subsample_pod5(pod5_path, read_name_path, output_directory):
    """
    Takes a POD5 file and a text file containing read names of interest and writes out a subsampled POD5 for just those reads.
    This is a useful function when you have a list of read names that mapped to a region of interest that you want to reanalyze from the pod5 level.

    Parameters:
        pod5_path (str): File path to the POD5 file (or directory of multiple pod5 files) to subsample.
        read_name_path (str | int): File path to a text file of read names. One read name per line. If an int value is passed, a random subset of that many reads will occur
        output_directory (str): A file path to the directory to output the file.

    Returns:
        None
    """
    import pod5 as p5
    import os

    if os.path.isdir(pod5_path):
        pod5_path_is_dir = True
        input_pod5_base = 'input_pod5s.pod5'
        files = os.listdir(pod5_path)
        pod5_files = [os.path.join(pod5_path, file) for file in files if '.pod5' in file]
        pod5_files.sort()
        print(f'Found input pod5s: {pod5_files}')
    
    elif os.path.exists(pod5_path):
        pod5_path_is_dir = False
        input_pod5_base = os.path.basename(pod5_path)

    else:
        print('Error: pod5_path passed does not exist')
        return None

    if type(read_name_path) == str:
        input_read_name_base = os.path.basename(read_name_path)
        output_base = input_pod5_base.split('.pod5')[0] + '_' + input_read_name_base.split('.txt')[0] + '_subsampled.pod5'

        # extract read names into a list of strings
        with open(read_name_path, 'r') as file:
            read_names = [line.strip() for line in file]

        print(f'Looking for read_ids: {read_names}')
        read_records = []

        if pod5_path_is_dir:
            for input_pod5 in pod5_files:
                with p5.Reader(input_pod5) as reader:
                    try:
                        for read_record in reader.reads(selection=read_names, missing_ok=True):
                            read_records.append(read_record.to_read())      
                            print(f'Found read in {input_pod5}: {read_record.read_id}')      
                    except:
                        print('Skipping pod5, could not find reads')    
        else:    
            with p5.Reader(pod5_path) as reader:
                try:
                    for read_record in reader.reads(selection=read_names):
                        read_records.append(read_record.to_read())
                        print(f'Found read in {input_pod5}: {read_record}')  
                except:
                    print('Could not find reads')   

    elif type(read_name_path) == int:
        import random
        output_base = input_pod5_base.split('.pod5')[0] + f'_{read_name_path}_randomly_subsampled.pod5'
        all_read_records = []

        if pod5_path_is_dir:
            # Shuffle the list of input pod5 paths
            random.shuffle(pod5_files)
            for input_pod5 in pod5_files:
                # iterate over the input pod5s
                print(f'Opening pod5 file {input_pod5}')
                with p5.Reader(pod5_path) as reader:
                    for read_record in reader.reads():
                        all_read_records.append(read_record.to_read())
                # When enough reads are in all_read_records, stop accumulating reads.
                if len(all_read_records) >= read_name_path:
                    break

            if read_name_path <= len(all_read_records):
                read_records = random.sample(all_read_records, read_name_path)
            else:
                print('Trying to sample more reads than are contained in the input pod5s, taking all reads')
                read_records = all_read_records
                    
        else:
            with p5.Reader(pod5_path) as reader:
                for read_record in reader.reads():
                    # get all read records from the input pod5
                    all_read_records.append(read_record.to_read())
            if read_name_path <= len(all_read_records):
                # if the subsampling amount is less than the record amount in the file, randomly subsample the reads
                read_records = random.sample(all_read_records, read_name_path)
            else:
                print('Trying to sample more reads than are contained in the input pod5s, taking all reads')
                read_records = all_read_records

    output_pod5 = os.path.join(output_directory, output_base)

    # Write the subsampled POD5
    with p5.Writer(output_pod5) as writer:
        writer.add_reads(read_records)