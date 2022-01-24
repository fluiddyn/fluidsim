#!/bin/bash
set -e

cd $WORK/Dev/fluidfft
hg pull
hg up default
python setup.py develop
# Does not work with (uses MPI, which is forbidden on the frontales)
# pip install -e . --no-build-isolation

cd $WORK/Dev/fluidsim
make cleanall
# --no-build-isolation to use pythran already installed in the environment
pip install -e . --no-build-isolation

pytest fluidsim

cd doc/examples/clusters/jean_zay/
