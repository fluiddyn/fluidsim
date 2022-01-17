#!/bin/bash
set -e

module purge
module load mercurial/6.0

cd $WORK/Dev
rm -rf fluiddyn fluidfft

hg clone https://foss.heptapod.net/fluiddyn/fluiddyn
hg clone https://foss.heptapod.net/fluiddyn/fluidfft
