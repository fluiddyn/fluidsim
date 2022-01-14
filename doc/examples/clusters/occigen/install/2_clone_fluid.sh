#!/bin/bash
set -e

mkdir -p $HOME/Dev
cd $HOME/Dev
rm -rf fluiddyn fluidfft fluidsim
hg clone ssh://hg@foss.heptapod.net/fluiddyn/fluiddyn
hg clone ssh://hg@foss.heptapod.net/fluiddyn/fluidfft
hg clone ssh://hg@foss.heptapod.net/fluiddyn/fluidsim
