## Let's start on Occigen

First, create a ssh key
(https://foss.heptapod.net/help/ssh/README#generating-a-new-ssh-key-pair) and
copy the public key in https://foss.heptapod.net and https://heptapod.host.

Download setup files:

```bash
wget https://heptapod.host/meige/milestone-sim/raw/branch/default/jean_zay/conf_files/setup_ssh.sh
```

Source these files:

```bash
. ~/setup_ssh.sh
```

load the mercurial environment and move to the work directory:

```bash
module load mercurial/6.0
cd $WORK 
```

You should be able to clone this repository without entering your password:

```bash
mkdir Dev
cd Dev
hg clone ssh://hg@heptapod.host/meige/milestone-sim
```

## Installation

See https://heptapod.host/meige/milestone-sim/tree/branch/default/jean_zay/install

```bash
cd milestone-sim/jean_zay/install
```

## Setup your environment and submit simulations

When you login into Occigen:

1. Source some files:

```bash
. ~/setup_hg.sh
. ~/setup_ssh.sh
. setup_env_full.sh
```

Then run the submit file, for example:

```bash
python scripts/submit_tests.py
```

### Useful commands for the SLURM scheduler

```bash
sbatch
squeue -u $USER
scancel
scontrol hold <job_list>
scontrol release <job_list>
scontrol show job $JOBID
find -maxdepth 1 -type d | while read -r dir; do printf "%s:\t" "$dir"; find "$dir" -type f | wc -l; done
```

## Install h5py parallel (to modify environments created before 24/11/2020)

Activate the conda environment and run:
```bash
conda uninstall h5py
export CC=mpicc
export HDF5_MPI="ON"
unset HDF5_LIBDIR
export LDFLAGS="-Wl,--no-as-needed"
pip install h5py --no-binary h5py
```

You can check that h5py parallel is correctly installed by running
```bash
python -c "import h5py; cfg = h5py.get_config(); print(cfg.mpi)"
```
