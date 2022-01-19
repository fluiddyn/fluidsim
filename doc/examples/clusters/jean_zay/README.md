## Let's start on Jean-Zay

First, create a ssh key
(https://foss.heptapod.net/help/ssh/README#generating-a-new-ssh-key-pair) and
copy the public key in https://foss.heptapod.net and https://heptapod.host.

Download setup files:

```bash
cd
wget https://foss.heptapod.net/fluiddyn/fluidsim/-/blob/topic/default/install-clusters/doc/examples/clusters/jean_zay/conf_files/setup_ssh.sh
```

Source these files:

```bash
. ~/setup_ssh.sh
```

Load the mercurial and python environments and move to the work directory:

```bash
module load mercurial/6.0 python/3.8.8
# (hg debuginstall -T "{pythonexe}") to check which python is used by hg: it could be different from 3.8.8
cd $WORK 
```

# TODO: Document the installation of p3dfft-2.7.6 in $WORK and modify conf_files/.fluidfft-site.cfg


Configure conda

```bash
conda activate base
pip install conda-app     
export PATH=$PATH:$HOME/.local/bin/  
mkdir $WORK/.conda   
ln -s $WORK/.conda $HOME
conda config --add channels conda-forge
```


You should be able to clone this repository without entering your password:

```bash
mkdir Dev
cd Dev
hg clone https://foss.heptapod.net/fluiddyn/fluidsim
# It should work to clone the packages with ssh. 
cd fluidsim
hg pull
hg up install-clusters # install-clusters should be replaced by default when merged (or we can remove this line)
```

## Installation

Move to the install directory and run the following commands (from this directory):

```bash
cd doc/examples/clusters/jean_zay/install
source 0_setup_env_base.sh
./1_copy_config_files.sh
source ~/setup_ssh.sh  # Is it necessary?
./2_clone_fluid.sh
./3_create_conda_env.sh
source 4_setup_env_conda.sh
./5_update_install_fluid.sh 
# cd .. && make  # does not work
```

**Note:** # TODO: put somes notes here if there are some troubles during the installation


## Submit the MPI test suite

Finally, you can submit the checks and tests using MPI by doing

```bash
cd ../scripts
python submit_check_fluidfft.py
python submit_tests.py
# TODO: "RuntimeError: Undefined plan with nthreads. This is a bug"
python submit_simul.py
# TODO: 
```

## Setup Mercurial

Correct and uncomment the line with the username and email address in
`~/.hgrc`:

```bash
vim ~/.hgrc
```

(see also
https://fluiddyn.readthedocs.io/en/latest/mercurial_heptapod.html#set-up-mercurial)

```bash
cd milestone-sim/jean_zay/install
```

## Setup your environment and submit simulations

When you login into Jean-Zay:

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

