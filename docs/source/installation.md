# Installation

## PyPi version

Pull smftools from [PyPI](https://pypi.org/project/smftools):

```shell
pip install smftools
```

It is recommended to first create and activate a conda environment before installing smftools to ensure dependencies are managed smoothly:

```shell
conda create -n smftools
conda activate smftools
pip install smftools
```

Ensure that you can access dorado, samtools, modkit, bedtools, and BedGraphtoBigWig executables from the terminal in this environment. These are all necessary for the functionality within the Informatics module.
You may need to add them to $PATH if they are not globally configured.
For example, if you want to check if dorado is executable, simply run this in the terminal:

```shell
dorado
```

On Mac OSX, the following can be used to congigure bedtools (with brew) and BedGraphToBigWig (with wget). Change the BedGraphToBigWig link to include the correct architecture for your OS.

```shell
brew install bedtools
wget http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/bedGraphToBigWig
chmod +x bedGraphToBigWig
sudo mv bedGraphToBigWig /usr/local/bin/
```

## Development Version

Clone smftools from source and change into the smftools directory:

```shell
git clone https://github.com/jkmckenna/smftools.git
cd smftools
```

A virtual environment can be created for the current version within the smftools directory:

```shell
python -m venv venv-smftools
source venv-smftools/bin/activate
pip install .
```

Subsequent use of the installed version of smftools can be run by changing to the smftools directory and activating the venv:

```shell
cd smftools
source venv-smftools/bin/activate
```

You can now run smftools from the terminal, an IDE, or a notebook within the virtual environment.