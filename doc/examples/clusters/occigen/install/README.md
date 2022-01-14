# Install

First, create a ssh key
(https://foss.heptapod.net/help/ssh/README#generating-a-new-ssh-key-pair) and
copy the public key in https://foss.heptapod.net and https://heptapod.host.

Then, run the following commands (from this directory):

```bash
source 0_setup_env_base.sh
./1_copy_config_files.sh
source ~/setup_ssh.sh
./2_clone_fluid.sh
./3_create_conda_env.sh
source 4_setup_env_conda.sh
./5_update_install_fluid.sh
cd .. && make
```

**Note:** you may experience an error like `remote: ssh: Could not resolve
hostname foss.heptapod.net: Temporary failure in name resolution`. Just retry
and it's going to work.

During the last step, some tests should be run, for me (Pierre), one
of the test fails (in
fluidsim/solvers/ad1d/pseudo_spect/test_solver.py) but it does not
seem to be a big problem.

## Submit the MPI test suite

Finally, you can submit the tests using MPI by doing

```bash
cd ..
python submit_tests.py
```

## Setup Mercurial

Correct and uncomment the line with the username and email address in
`~/.hgrc`:

```bash
vim ~/.hgrc
```

(see also
https://fluiddyn.readthedocs.io/en/latest/mercurial_heptapod.html#set-up-mercurial)
