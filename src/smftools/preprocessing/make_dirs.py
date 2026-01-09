## make_dirs

from smftools.logging_utils import get_logger

logger = get_logger(__name__)

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
            logger.info("Directory '%s' created successfully.", directory)
        else:
            logger.info("Directory '%s' already exists.", directory)
