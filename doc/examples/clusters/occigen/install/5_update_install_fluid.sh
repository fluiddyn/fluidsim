#!/bin/bash
set -e

cd $HOME/Dev/fluiddyn
hg pull
hg up default
make clean
pip install -e .

cd $HOME/Dev/fluidfft
hg pull
hg up default
make cleanall
pip install -e .

# this should work!
python -c "import fluidfft.fft3d.mpi_with_p3dfft"
python -c "import fluidfft.fft3d.mpi_with_pfft"

cd $HOME/Dev/fluidsim
hg pull
hg up default
make cleanall
pip install -e .

pytest fluidsim
