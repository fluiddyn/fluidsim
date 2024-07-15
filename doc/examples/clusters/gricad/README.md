# Using Fluidsim on Gricad clusters

We show how to use Fluidsim on Gricad clusters. The main documentation for this
HPC platform is [here]( https://gricad-doc.univ-grenoble-alpes.fr/hpc/). We
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

```sh
# This will take a while
# note: pip does not work with the -C (container) option
guix shell -m ~/dev/fluidsim/doc/examples/clusters/gricad/manifest.scm

python3 -m venv ~/venv-fluidsim-guix --system-site-packages
. ~/venv-fluidsim-guix/bin/activate
cd ~/dev/fluidsim
pip install -e lib
pip install -e ".[test]" -v --config-settings=setup-args=-Dnative=true --no-build-isolation
```

## Test Fluidsim in sequential

```sh
source /applis/site/guix-start.sh
guix shell -m ~/dev/fluidsim/doc/examples/clusters/gricad/manifest.scm
. ~/venv-fluidsim-guix/bin/activate
cd ~/dev/fluidsim
python -m pytest fluidsim
```

## Submit a fluidfft benchmark

```sh
cd ~/dev/fluidsim/doc/examples/clusters/gricad/
oarsub -S ./job_fluidfft_bench.oar
```
