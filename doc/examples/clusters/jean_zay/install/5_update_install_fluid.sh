#!/bin/bash
set -e

cd $WORK/Dev/fluiddyn
hg pull
hg up cluster-jean-zay # cluster-jean-zay should be replaced by default when merged 
make clean
pip install -e .

cd $WORK/Dev/fluidfft
hg pull
hg up default 
cp $WORK/Dev/fluidsim/doc/examples/clusters/jean_zay/conf_files/.fluidfft-site.cfg site.cfg
# pip install -e .   seems to run something with mpi, which is forbidden 
python setup.py develop

cd $WORK/Dev/fluidsim
make cleanall
# pip install -e .
python setup.py develop

pytest fluidsim
