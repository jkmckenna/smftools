## make_dirs

# General
def make_dirs(directories):
    """
    Takes a list of file paths and makes new directories if the directory does not already exist.

    Parameters:
        directories (list): A list of directories to make
    
    Returns:
        None
    """
    import os

    for directory in directories:
        if not os.path.isdir(directory):
            os.mkdir(directory)
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Directory '{directory}' already exists.")