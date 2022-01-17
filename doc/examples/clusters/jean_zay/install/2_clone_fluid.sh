#!/bin/bash
set -e

cd $WORK/Dev
rm -rf fluiddyn fluidfft fluidsim
hg clone https://foss.heptapod.net/fluiddyn/fluiddyn
hg clone https://foss.heptapod.net/fluiddyn/fluidfft
hg clone https://foss.heptapod.net/fluiddyn/fluidsim
