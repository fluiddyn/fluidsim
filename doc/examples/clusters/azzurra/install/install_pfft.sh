cd $HOME

git clone https://github.com/mpip/pfft

OPT=$HOME/opt
ROOTFFTW3=/softs/fftw
ROOTPFFT=$OPT/pfft

mkdir -p $ROOTPFFT

cd pfft
export LANG=C
./bootstrap.sh
./configure --prefix=$ROOTPFFT --with-fftw3=$ROOTFFTW3 CC=mpicc CCLD='mpif90 -nofor_main'

make install
