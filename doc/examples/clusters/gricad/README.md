# Using Fluidsim on Gricad clusters

We show how to use Fluidsim on Gricad clusters. The main documentation for this
HPC platform is [here](https://gricad-doc.univ-grenoble-alpes.fr/hpc/). We
will use [Guix](https://gricad-doc.univ-grenoble-alpes.fr/hpc/softenv/guix/),
which is one of the recommended package managers for this platform.

## Setup Guix

The first thing to do, is to create the following file
`~/.config/guix/channels.scm`:

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

> You only need to update the guix environment (and thus run `guix pull`) when
a package you want to use has been created or updated.

After `guix pull`, you should run the following command to be sure you use the
lastest `guix` command:

```sh
GUIX_PROFILE="$HOME/.config/guix/current"
. "$GUIX_PROFILE/etc/profile"
```

## Install Fluidsim from source

Clone the Fluidsim repository in `$HOME/dev`.

```sh
source /applis/site/guix-start.sh
DIR_MANIFEST=$HOME/dev/fluidsim/doc/examples/clusters/gricad
# This will take a while
guix shell --pure -m $DIR_MANIFEST/manifest.scm -f $DIR_MANIFEST/python-fluidsim.scm
```

### Change the changeset used for the Guix environment

One can choose another changeset reference. One can study them with:

```sh
cd ~/dev/fluidsim
hg log -G
```

Get the hash with

```sh
# Actually it does not work (no support for Mercurial)
# How should we obtain the sha256 base32 hash?
# guix download -r --commit=changeset_ref https://foss.heptapod.net/fluiddyn/fluidsim
```

```sh
source /applis/site/guix-start.sh
hg clone https://foss.heptapod.net/fluiddyn/fluidsim
cd ~/dev/fluidsim-clean
hg up <changeset_ref> --clean
hg purge --all
guix hash -x -r .
```

Change the Mercurial reference and the hash in `python-fluidsim.scm`.

## Test Fluidsim in sequential

```sh
source /applis/site/guix-start.sh
DIR_MANIFEST=$HOME/dev/fluidsim/doc/examples/clusters/gricad
guix shell --pure -m $DIR_MANIFEST/manifest.scm -f $DIR_MANIFEST/python-fluidsim.scm
python3 -m pytest --pyargs fluidsim
```

## Submit a Fluidfft benchmark

```sh
source /applis/site/guix-start.sh
cd ~/dev/fluidsim/doc/examples/clusters/gricad
oarsub -S ./job_fluidfft_bench.oar
```

## Submit a Fluidsim benchmark

```sh
source /applis/site/guix-start.sh
cd ~/dev/fluidsim/doc/examples/clusters/gricad
oarsub -S ./job_fluidsim_bench.oar
```

or

```sh
cd ~/dev/fluidsim/doc/examples/clusters/gricad
/usr/bin/python3 -m venv .venv
. .venv/bin/activate
pip install fluiddyn
python3 submit_bench_fluidsim.py
```
