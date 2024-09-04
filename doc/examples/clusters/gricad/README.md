# Using Fluidsim on Gricad clusters

We show in this directory
(<https://foss.heptapod.net/fluiddyn/fluidsim/-/tree/branch/default/doc/examples/clusters/gricad>)
how to use Fluidsim on Gricad clusters. The main documentation for this HPC platform is
[here](https://gricad-doc.univ-grenoble-alpes.fr/hpc/). We will use
[Guix](https://gricad-doc.univ-grenoble-alpes.fr/hpc/softenv/guix/), which is one of the
recommended package managers for this platform.

## Get a login and setup ssh

Get an account on <https://perseus.univ-grenoble-alpes.fr/>.

Set an ssh key and the alias

```sh
alias sshdahu='ssh -X dahu.ciment'
```

## Setup Guix

The first thing to do, is to create the following file `~/.config/guix/channels.scm`:

```lisp
(cons* (channel
          (name 'gricad-guix-packages)
          (url "https://gricad-gitlab.univ-grenoble-alpes.fr/bouttiep/gricad_guix_packages.git")
          (branch "WIP_Benjamin"))
       %default-channels)
```

Once this is done, you can load and update the Guix environment:

```sh
source /applis/site/guix-start.sh
guix pull  # This will take a while
```

You only need to update the guix environment (and thus run `guix pull`) when a package
you want to use has been created or updated.

After `guix pull`, you should run the following command to be sure you use the lastest
`guix` command:

```sh
GUIX_PROFILE="$HOME/.config/guix/current"
. "$GUIX_PROFILE/etc/profile"
```

## Install Fluidsim from source

Install and setup Mercurial as explained
[here](https://fluidhowto.readthedocs.io/en/latest/mercurial/install-setup.html). Clone
the Fluidsim repository in `$HOME/dev`.

### Change the changeset used for the Guix environment

One needs to choose a changeset reference (a hash). One can study them with:

```sh
cd ~/dev/fluidsim
hg log -G
```

Get the hash with (note: `guix download` does not support Mercurial):

```sh
source /applis/site/guix-start.sh
hg clone https://foss.heptapod.net/fluiddyn/fluidsim ~/dev/fluidsim-clean
cd ~/dev/fluidsim-clean
hg up <changeset_ref> --clean
hg purge --all
guix hash -x -r .
```

Change the Mercurial reference and the hash in `python-fluidsim.scm`.

### Build-install from source

```sh
source /applis/site/guix-start.sh
DIR_MANIFEST=$HOME/dev/fluidsim/doc/examples/clusters/gricad
# This will take a while
guix shell --pure -m $DIR_MANIFEST/manifest.scm -f $DIR_MANIFEST/python-fluidsim.scm
```

## Test Fluidsim in sequential

```sh
source /applis/site/guix-start.sh
DIR_MANIFEST=$HOME/dev/fluidsim/doc/examples/clusters/gricad
guix shell --pure -m $DIR_MANIFEST/manifest.scm -f $DIR_MANIFEST/python-fluidsim.scm
python3 -m pytest --pyargs fluidsim
```

## Submit a Fluidfft benchmark

```sh
cd ~/dev/fluidsim/doc/examples/clusters/gricad
oarsub -S ./job_fluidfft_bench.oar
```

## Submit a Fluidsim benchmark

### Hand written OAR script

```sh
cd ~/dev/fluidsim/doc/examples/clusters/gricad
oarsub -S ./job_fluidsim_bench.oar
```

### With fluiddyn

Prepare a virtual env (1 time).

```sh
source /applis/environments/conda.sh
conda create -n env-fluiddyn fluiddyn
```

Submit with

```sh
cd ~/dev/fluidsim/doc/examples/clusters/gricad
source /applis/environments/conda.sh
conda activate env-fluiddyn
python submit_bench_fluidsim.py
```
