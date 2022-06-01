#!/bin/bash
set -e

unset CC
unset CXX

cd $HOME/Dev

# Install fluidfft from sources
cd fluidfft
make clean
hg up default
python setup.py develop 
#pip install -e . --no-build-isolation 
#make

# Install fluidsim from sources
cd ../fluidsim
make cleanall
pip install -e . --no-build-isolation

pytest fluidsim

cd $HOME/Dev/fluidsim/doc/examples/clusters/azzurra/
