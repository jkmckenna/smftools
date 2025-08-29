import click
from . import load_adata

@click.group()
def cli():
    """Command-line interface for smftools."""
    pass

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def load(config_path):
    """Load and process data from CONFIG_PATH."""
    load_adata(config_path)
