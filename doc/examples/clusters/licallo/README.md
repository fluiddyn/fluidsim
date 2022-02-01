## Let's start on Licallo

### SSH key

First, create a ssh key
(https://foss.heptapod.net/help/ssh/index#generate-an-ssh-key-pair) and
copy the public key in https://foss.heptapod.net.

### Download setup files

Download and source setup file:

```bash
cd $HOME
# TODO: Modify the urls before merging
wget https://heptapod.host/meige/milestone-sim/raw/branch/default/occigen/conf_files/setup_hg.sh --no-check-certificate
. ~/setup_ssh.sh
wget https://foss.heptapod.net/fluiddyn/fluidsim/-/blob/branch/default/doc/examples/clusters/occigen/conf_files/.hgrc --no-check-certificate
```

### Load the python module and create a new environment with mercurial

```bash
module purge
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda config --add channels conda-forge
conda activate base 
pip install conda-app
conda-app install mercurial
conda run -n _env_mercurial pip install hg-fluiddyn
```

### Configure mercurial:

Correct and uncomment the line with the username and email address in ~.hgrc (see also
https://fluiddyn.readthedocs.io/en/latest/mercurial_heptapod.html#set-up-mercurial)

```bash
nano ~/.hgrc
```

### Clone the fluidsim repository

```bash
mkdir Dev
cd Dev	
hg clone https://foss.heptapod.net/fluiddyn/fluidsim
cd fluidsim
hg up install-clusters-licallo # TODO: remove this line before merging
```

### Install p3dfft-2.7.6 in your $HOME directory

We configured the installation of p3dfft-2.7.6 in $HOME such that you simply have to run a bash script:

```bash
source $HOME/Dev/fluidsim/doc/examples/clusters/licallo/setup_env_base.sh
bash $HOME/Dev/fluidsim/doc/examples/clusters/licallo/install/install_p3dfft.sh
TODO: Write install_p3dfft.sh p3dfft (see https://fluidfft.readthedocs.io/en/latest/install/occigen.html)
```

## Installation

Move to the install directory and run the following commands (from this directory):

```bash
cd $HOME/Dev/fluidsim/doc/examples/clusters/licallo/install
source 0_setup_env_base.sh
./1_copy_config_files.sh
source ~/setup_ssh.sh 
./2_create_conda_env.sh
source 3_setup_env_conda.sh
./4_clone_fluid.sh
./5_update_install_fluid.sh
cd .. && make
```

## Submit the MPI test suite

Finally, you can submit the checks and tests using MPI by doing

```bash
cd ../scripts
python submit_check_fluidfft.py
python submit_tests.py
python submit_simul.py
```


## Setup your environment and submit simulations

When you login into Licallo:

1. Source some files:

```bash
cd $WORK/Dev/fluidsim/doc/examples/clusters/licallo
. ~/setup_ssh.sh
. setup_env_full.sh
```

Then run the submit file, for example:

```bash
python scripts/submit_simul.py
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
