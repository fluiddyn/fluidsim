cd $WORK/Dev
rm -rf fluiddyn fluidfft

hg clone https://foss.heptapod.net/fluiddyn/fluiddyn
hg clone https://foss.heptapod.net/fluiddyn/fluidfft
hg clone https://foss.heptapod.net/fluiddyn/transonic # TODO: remove this line when topic fix-bug-mpi-barrier-jean-zay is merged


cd $WORK/Dev/fluidsim/doc/examples/clusters/jean_zay/install
