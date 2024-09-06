## load_experiment_config

def load_experiment_config(experiment_config):
    """
    Loads in the experiment configuration csv and saves global variables with experiment configuration parameters.
    Parameters:
        experiment_config (str): A string representing the file path to the experiment configuration csv file.
    
    Returns:
        None
    """
    import csv

    with open(experiment_config, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract variable name and value from each row
            var_name = row['variable']
            value = row['value']

            # Alternatively, set it directly in the globals() dictionary
            globals()[var_name] = value

