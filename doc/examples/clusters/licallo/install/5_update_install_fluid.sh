#!/bin/bash
set -e

unset CC
unset CXX

cd $HOME/Dev

# Install fluidfft from sources
cd fluidfft
hg up default
# issue with MPI not allowed on the frontale
#pip install -e .
python setup.py develop

# Install fluidsim from sources
cd ../fluidsim
make cleanall
pip install -e . --no-build-isolation

pytest fluidsim

cd $HOME/Dev/fluidsim/doc/examples/clusters/licallo/
