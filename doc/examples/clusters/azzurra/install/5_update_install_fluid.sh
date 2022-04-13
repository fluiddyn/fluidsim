#!/bin/bash
set -e

unset CC
unset CXX

cd $HOME/Dev

# TODO: Remove when the topic is merged
# Install fluiddyn from sources
cd fluiddyn
hg up install_azzurra
pip install -e . --no-build-isolation

# Install fluidfft from sources
cd ../fluidfft
hg up default
python setup.py develop

# Install fluidsim from sources
cd ../fluidsim
make cleanall
pip install -e . --no-build-isolation

pytest fluidsim

cd $HOME/Dev/fluidsim/doc/examples/clusters/azzurra/
