cd $HOME

git clone https://github.com/mpip/pfft

OPT=$HOME/opt
ROOTFFTW3=/opt/ohpc/pub/oca/apps/intel2020u2-gnu8/impi/fftw3/3.3.8
ROOTPFFT=$OPT/pfft

mkdir -p $ROOTPFFT

cd pfft
export LANG=C
./bootstrap.sh
./configure --prefix=$ROOTPFFT --with-fftw3=$ROOTFFTW3 CC=mpicc CCLD='mpif90 -nofor_main'

make install
