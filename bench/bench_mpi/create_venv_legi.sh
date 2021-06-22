# This has to be run on a node of a cluster at LEGI.
# From the frontale, run:
#
# oarsub -I -l "{net='ib' and os='buster'}/core=2,walltime=2:00:0"

SHARED=/fsnet/project/watu/2020/20MILESTONE/Bardant

module purge
module load python/3.8.0 openmpi/4.0.5-ib

mkdir -p $SHARED/envs

python -m venv $SHARED/envs/env_fluidsim

source $SHARED/envs/env_fluidsim/bin/activate

pip install pip -U
pip install git+https://github.com/mpi4py/mpi4py.git --no-binary mpi4py
pip install hg+https://foss.heptapod.net/fluiddyn/fluiddyn

cd ~/Dev/fluidfft
make

cd ~/Dev/fluidsim
make

# Afterwards, the environment needs to be activated with
#
# source $SHARED/envs/env_fluidsim/bin/activate
