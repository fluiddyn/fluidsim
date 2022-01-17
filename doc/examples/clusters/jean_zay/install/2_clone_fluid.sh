#!/bin/bash
set -e

cd $WORK/Dev
rm -rf fluiddyn fluidfft

hg clone https://foss.heptapod.net/fluiddyn/fluiddyn
hg clone https://foss.heptapod.net/fluiddyn/fluidfft
