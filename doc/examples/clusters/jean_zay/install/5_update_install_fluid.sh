#!/bin/bash
set -e

cd $WORK/Dev/fluiddyn
hg pull
hg up cluster-jean-zay # cluster-jean-zay should be replaced by default when merged
make clean
pip install -e .


cd $WORK/Dev/transonic
hg pull
hg up default
make clean
pip install -e .


cd $WORK/Dev/fluidfft
hg pull
hg up default
python setup.py develop
# TODO: QUESTION for Vincent: does this work for fluidfft?
# pip install -e . --no-build-isolation
# Vincent: No

cd $WORK/Dev/fluidsim
make cleanall
# --no-build-isolation to use pythran already installed in the environment
pip install -e . --no-build-isolation
pytest fluidsim

cd doc/examples/clusters/jean_zay/
