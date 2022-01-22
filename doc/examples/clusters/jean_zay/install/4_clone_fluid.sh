cd $WORK/Dev
rm -rf fluiddyn fluidfft

hg clone https://foss.heptapod.net/fluiddyn/fluiddyn
hg clone https://foss.heptapod.net/fluiddyn/fluidfft
hg clone https://foss.heptapod.net/fluiddyn/transonic

cd $WORK/Dev/fluidsim/doc/examples/clusters/jean_zay/install
