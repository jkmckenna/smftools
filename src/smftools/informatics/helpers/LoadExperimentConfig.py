## LoadExperimentConfig

class LoadExperimentConfig:
    """
    Loads in the experiment configuration csv and saves global variables with experiment configuration parameters.
    Parameters:
        experiment_config (str): A string representing the file path to the experiment configuration csv file.
    
    Attributes:
        var_dict (dict): A dictionary containing experiment configuration parameters.

    Example:
        >>> import pandas as pd
        >>> from io import StringIO
        >>> csv_data = '''variable,value,type
        ... mapping_threshold,0.05,float
        ... batch_size,4,int
        ... testing_bool,True,bool
        ... strands,"[bottom, top]",list
        ... split_dir,split_bams,string
        ... pod5_dir,None,string
        ... pod5_dir,,string
        ... '''
        >>> csv_file = StringIO(csv_data)
        >>> df = pd.read_csv(csv_file)
        >>> df.to_csv('test_config.csv', index=False)
        >>> config_loader = LoadExperimentConfig('test_config.csv')
        >>> config_loader.var_dict['mapping_threshold']
        0.05
        >>> config_loader.var_dict['batch_size']
        4
        >>> config_loader.var_dict['testing_bool']
        True
        >>> config_loader.var_dict['strands']
        ['bottom', 'top']
        >>> config_loader.var_dict['split_dir']
        'split_bams'
        >>> config_loader.var_dict['pod5_dir'] is None
        True
        >>> config_loader.var_dict['pod5_dir'] is None
        True
    """
    def __init__(self, experiment_config):
        import pandas as pd
        print(f"Loading experiment config from {experiment_config}")
        # Read the CSV into a pandas DataFrame
        df = pd.read_csv(experiment_config)
        # Initialize an empty dictionary to store variables
        var_dict = {}
        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            var_name = str(row['variable'])
            value = row['value']
            dtype = row['type']
            # Handle empty and None values
            if pd.isna(value) or value in ['None', '']:
                value = None
            else:
                # Handle different data types
                if dtype == 'list':
                    # Convert the string representation of a list to an actual list
                    value = value.strip('()[]').replace(', ', ',').split(',')
                elif dtype == 'int':
                    value = int(value)
                elif dtype == 'float':
                    value = float(value)
                elif dtype == 'bool':
                    value = value.lower() == 'true'
                elif dtype == 'string':
                    value = str(value)
            # Store the variable in the dictionary
            var_dict[var_name] = value
        # Save the dictionary as an attribute of the class
        self.var_dict = var_dict

