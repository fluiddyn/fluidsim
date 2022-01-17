#!/bin/bash
set -e

cp $WORK/Dev/milestone-sim/jean_zay/conf_files/.pythranrc ~/.pythranrc
cp $WORK/Dev/milestone-sim/jean_zay/conf_files/.hgrc ~/.hgrc

cd $WORK/Dev/fluiddyn
hg pull
hg up default
make clean
pip install -e .

cd $WORK/Dev/fluidfft
hg pull
hg up default
cp $WORK/Dev/milestone-sim/jean_zay/conf_files/.fluidfft-site.cfg site.cfg
python setup.py develop

# this should work!
python -c "import fluidfft.fft3d.mpi_with_fftw3d"
python -c "import fluidfft.fft3d.mpi_with_p3dfft"

cd $WORK/Dev/fluidsim
hg pull
hg up default
make cleanall
pip install -e .

pytest fluidsim
