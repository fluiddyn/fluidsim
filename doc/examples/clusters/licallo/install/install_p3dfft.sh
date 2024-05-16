cd $HOME

git clone https://github.com/CyrilleBonamy/p3dfft.git

OPT=$HOME/opt
ROOTFFTW3=/opt/ohpc/pub/oca/apps/intel2020u2-gnu8/impi/fftw3/3.3.8
ROOTP3DFFT=$OPT/p3dfft/2.7.5

mkdir -p $ROOTP3DFFT

CC=mpicc
FC=mpif90

cd p3dfft

libtoolize && aclocal && autoconf && automake --add-missing

./configure --enable-fftw --with-fftw=$ROOTFFTW3 --prefix=$ROOTP3DFFT \
    CC=${CC} CCLD=${FC}  

make install
