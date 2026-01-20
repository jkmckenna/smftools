# Installation

## PyPi version - Easiest starting point

It is recommended to first create and activate a conda environment before installing smftools to ensure dependencies are managed smoothly. The following sets up a conda environment, activates the environment, and installs the base smftools and uses a Pytorch backend:

```shell
conda create -n smftools
conda activate smftools
pip install --upgrade pip
pip install smftools[torch]
```

Ensure that you can access dorado, modkit, and minimap2 executables from the terminal in this environment.
You may need to add them to $PATH if they are not globally configured.
For example, if you want to check if dorado is executable, simply run this in the terminal:

```shell
dorado
```

On Mac OSX, the following can be used to congigure minimap2 (with brew) if you do not want to use the dorado aligner.
Also, optional samtools (brew), bedtools (brew) and BedGraphToBigWig (with wget) can be installed and configured if you prefer to use the CLI backend. SMFtools also has the option to use a Python backend instead of these CLI backends if you pip install with the optional dependency flags: pysamtools, pybedtools, pybigwig.

```shell
brew install minimap2
wget http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/bedGraphToBigWig
chmod +x bedGraphToBigWig
sudo mv bedGraphToBigWig /usr/local/bin/
```

If you are starting from Nanopore POD5 files, you will need to install with the ont option.
```shell
pip install smftools[ont]
```

If you want multiqc to be performed on your experiment, you will need to use the qc option on install.
```shell
pip install smftools[qc]
```

If you want to use the automated plotting functions, you will need to install the plotting dependencies.
```shell
pip install smftools[plotting]
```

If you prefer to use python backends (or do not have the CLI backends installed) for samtools, bedtools, BedGraphToBigWig, use the following options.
```shell
pip install smftools[pysamtools,pybedtools,pybigwig]
```

If you want to generate UMAPs and perform Leiden clustering, use the following options.
```shell
pip install smftools[scanpy,clustering]
```

If you just want to install the full functionality of smftools, just use the following install option.
```shell
pip install smftools[all]
```

## Development Version - recommended to use this method for most up to date versions

Clone smftools from source and change into the smftools directory:

```shell
git clone https://github.com/jkmckenna/smftools.git
cd smftools
```

A python virtual environment can be created as an alternative to conda. I like to do venv-smftools-X.X.X to keep a seperate venv for each version:

```shell
python -m venv venv-smftools
source venv-smftools/bin/activate
pip install --upgrade pip
pip install .
pip install ipykernel jupyter
python -m ipykernel install --user --name=venv-smftools --display-name "Python (smftools)"
```

Subsequent use of the installed version of smftools can be run by changing to the smftools directory and activating the venv:

```shell
cd smftools
source venv-smftools/bin/activate
```

You can now run smftools from the terminal, an IDE, or a notebook within the virtual environment.