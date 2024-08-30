## make_dirs
import os

# General
def make_dirs(directories):
    """
    Input: Takes a list of file paths to make directories for
    Output: Makes each directory in the list if the directory doesn't already exist.
    """
    for directory in directories:
        if not os.path.isdir(directory):
            os.mkdir(directory)
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Directory '{directory}' already exists.")