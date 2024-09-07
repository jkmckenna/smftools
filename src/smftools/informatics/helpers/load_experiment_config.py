## load_experiment_config

class load_experiment_config:
    """
    Loads in the experiment configuration csv and saves global variables with experiment configuration parameters.
    Parameters:
        experiment_config (str): A string representing the file path to the experiment configuration csv file.
    
    Returns:
        None
    """
    def __init__(self, experiment_config):
        import csv
        with open(experiment_config, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            var_dict = {}
            for row in reader:
                # Extract variable name and value from each row
                var_name = row['variable']
                value = row['value']
                dtype = row['type']
                var_dict[var_name] = value
        self.var_dict = var_dict

