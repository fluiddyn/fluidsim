## Let's start on Jean-Zay

### SSH key

First, create a ssh key
(https://foss.heptapod.net/help/ssh/index#generate-an-ssh-key-pair) and
copy the public key in https://foss.heptapod.net.

### Download setup files

Download and source setup file:

```bash
cd $HOME
wget https://foss.heptapod.net/fluiddyn/fluidsim/-/raw/topic/default/install-clusters/doc/examples/clusters/jean_zay/conf_files/setup_ssh.sh
. ~/setup_ssh.sh
wget https://foss.heptapod.net/fluiddyn/fluidsim/-/raw/topic/default/install-clusters/doc/examples/clusters/jean_zay/conf_files/.hgrc
```

### Clone the fluidsim repository

Load the mercurial environment, move to the work directory and clone the fluidsim repository:

```bash
module load mercurial/6.0
mkdir -p $WORK/Dev
cd $WORK/Dev
# TODO: It should work to clone the packages with ssh. It should be tested.
hg clone https://foss.heptapod.net/fluiddyn/fluidsim
cd fluidsim
hg up install-clusters # TODO: remove this line before merging
module purge
```

**Note:** Mercurial will be installed inside the conda environment for fluidsim, so we will not use the mercurial module later.

### Configure conda

```bash
module load python/3.8.8
conda activate base
mkdir $WORK/.conda
ln -s $WORK/.conda $HOME
conda config --add channels conda-forge
```

### Install p3dfft-2.7.6 in your $WORK directory

We configured the installation of p3dfft-2.7.6 in $WORK such that you simply have to run a bash script:

```bash
module load automake/1.16.1 libtool/2.4.6
bash $WORK/Dev/fluidsim/doc/examples/clusters/jean_zay/install/install_p3dfft.sh
```

## Installation

Move to the install directory and run the following commands (from this directory):

```bash
cd $WORK/Dev/fluidsim/doc/examples/clusters/jean_zay/install
source 0_setup_env_base.sh
./1_copy_config_files.sh
source ~/setup_ssh.sh  # Is it necessary?
./2_create_conda_env.sh
source 3_setup_env_conda.sh
./4_clone_fluid.sh
./5_update_install_fluid.sh
cd .. && make
```

**Note:** # TODO: put some notes here if there are some troubles during the installation

## Submit the MPI test suite

Finally, you can submit the checks and tests using MPI by doing

```bash
cd ../scripts
python submit_check_fluidfft.py
python submit_tests.py
python submit_simul.py
```

## Setup Mercurial

Correct and uncomment the line with the username and email address in
`~/.hgrc`:

```bash
vim ~/.hgrc
```

(see also
https://fluiddyn.readthedocs.io/en/latest/mercurial_heptapod.html#set-up-mercurial)

## Setup your environment and submit simulations

When you login into Jean-Zay:

1. Source some files:

```bash
cd $WORK/Dev/fluidsim/doc/examples/clusters/jean_zay
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
squeue -u $USER | grep * | awk '{print $1}' | xargs scancel
scontrol hold <job_list>
scontrol release <job_list>
scontrol show job $JOBID
find -maxdepth 1 -type d | while read -r dir; do printf "%s:\t" "$dir"; find "$dir" -type f | wc -l; done
```
