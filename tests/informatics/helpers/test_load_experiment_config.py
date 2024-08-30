## load_experiment_config
import csv

def load_experiment_config(experiment_config):
    """
    Loads in the experiment configuration csv and saves global variables with experiment configuration parameters
    """
    with open(experiment_config, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract variable name and value from each row
            var_name = row['variable']
            value = row['value']

            # Alternatively, set it directly in the globals() dictionary
            globals()[var_name] = value

