cd $HOME

git clone https://github.com/CyrilleBonamy/p3dfft.git

OPT=$HOME/opt
ROOTFFTW3=/softs/fftw
ROOTP3DFFT=$OPT/p3dfft/2.7.5

mkdir -p $ROOTP3DFFT
cd p3dfft

libtoolize && aclocal && autoconf && automake --add-missing

./configure --enable-fftw --with-fftw=$ROOTFFTW3 --prefix=$ROOTP3DFFT \
    CC=mpicc CCLD=mpif90  

make install
