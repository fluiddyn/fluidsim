
alias duh1='du -h --max-depth=1'

module load intel openmpi hdf5

OPT=$SHAREDSCRATCHDIR/opt

# MINICONDA=$OPT/miniconda3
# MINICONDA=/home/augier/miniconda3
export MINICONDA=$SCRATCHDIR/../augier/miniconda3

export PATH=$MINICONDA/condabin:$PATH

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('$MINICONDA/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$MINICONDA/etc/profile.d/conda.sh" ]; then
        . "$MINICONDA/etc/profile.d/conda.sh"
    else
        export PATH="$MINICONDA/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPT/intel/pfft/lib:$OPT/intel/p3dfft/2.7.5/lib:$OPT/intel/fftw/3.3.7/lib
export LIBRARY_PATH=$LIBRARY_PATH:$OPT/intel/fftw/3.3.7/lib
export CPATH=$CPATH:$OPT/intel/fftw/3.3.7/include

# line added by conda-app
export PATH=$MINICONDA/condabin/app:$PATH

# needed to use clang for Pythran
unset CC
unset CXX

export FLUIDSIM_PATH=$SCRATCHDIR
