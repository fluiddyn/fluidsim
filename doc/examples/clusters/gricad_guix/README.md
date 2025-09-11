# Using Fluidsim on Gricad clusters

We show in this directory
(<https://foss.heptapod.net/fluiddyn/fluidsim/-/tree/branch/default/doc/examples/clusters/gricad_guix>)
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

## Setup Mercurial with UV

Install UV with

```sh
wget -qO- https://astral.sh/uv/install.sh | sh
```

Logout and login to get a new shell. UV should be available.

```sh
uv --version
```

Install and setup Mercurial:

```sh
uv tool install -p 3.13 mercurial --with hg-evolve --with hg-git
uvx hg-setup init -f
```

## Setup Guix

The first thing to do, is to create the file `~/.config/guix/channels.scm` with the
following content:

```lisp
(cons* (channel
          (name 'gricad-guix-packages)
          (url "https://gricad-gitlab.univ-grenoble-alpes.fr/bouttiep/gricad_guix_packages.git")
          (branch "master"))
       %default-channels)
```

Once this is done, you can load and update the Guix environment:

```sh
source /applis/site/guix-start.sh
guix pull  # This can take a very long time
```

You only need to update the Guix environment (and thus run `guix pull`) when a package
you want to use has been created or updated.

After `guix pull`, you have to run the following command to be sure you use the latest
`guix` command:

```sh
GUIX_PROFILE="$HOME/.config/guix/current"
. "$GUIX_PROFILE/etc/profile"
```

## Install Fluidsim from source

Clone the Fluidsim repository in `$HOME/dev`:

```sh
hg clone https://foss.heptapod.net/fluiddyn/fluidsim ~/dev/fluidsim
```

Update to

```sh
cd ~/dev/fluidsim
hg up doc-gricad-guix
```

### Change the changeset used for the Guix environment

One needs to choose a changeset (a commit) and get its changeset reference (its hash).
One can study them with:

```sh
cd ~/dev/fluidsim
# get the node (changeset reference, hash) of the current commit
# (you can choose this commit)
hg log -r . -T "{node}"
# study all commits
# (you can choose another commit)
hg log -G
```

Get the Guix hash with (note: `guix download` does not support Mercurial). In a new
terminal (replace `<changeset_ref>` with the chosen Mercurial node):

```sh
source /applis/site/guix-start.sh
hg clone https://foss.heptapod.net/fluiddyn/fluidsim ~/dev/fluidsim-clean
cd ~/dev/fluidsim-clean
hg up <changeset_ref> --clean
hg purge --all
guix hash -x -r .
```

Change the Mercurial reference and the hash in
`~/dev/fluidsim/doc/examples/clusters/gricad_guix/python-fluidsim.scm`.

### Build-install from source

```sh
source /applis/site/guix-start.sh
DIR_MANIFEST=$HOME/dev/fluidsim/doc/examples/clusters/gricad_guix/scm-files
# This will take a while
guix package -f $DIR_MANIFEST/python-fluidsim.scm --manifest=$DIR_MANIFEST/manifest.scm --profile=$HOME/guix-profile-fluidsim
```

It can be useful to list the package installed in the profile by:

``sh
guix package --list-installed --profile=$HOME/guix-profile-fluidsim
```

## Test Fluidsim in sequential

```sh
source $HOME/guix-profile-fluidsim/etc/profile
python -m pytest --pyargs fluidsim
```

## Submit a Fluidfft benchmark

```sh
ssh dahu-oar3
cd ~/dev/fluidsim/doc/examples/clusters/gricad_guix
source $HOME/guix-profile-fluidsim/etc/profile
oarsub -S ./job_fluidfft_bench.oar
```

## Submit a Fluidsim benchmark

Here, we are going to show how to do it with two strategies, either manually write a OAR
script or use fluiddyn to write it.

### Hand written OAR script

```sh
ssh dahu-oar3
cd ~/dev/fluidsim/doc/examples/clusters/gricad_guix
source $HOME/guix-profile-fluidsim/etc/profile
oarsub -S ./job_fluidsim_bench.oar
```

### With fluiddyn

Prepare a virtual env (1 time). From a new terminal:

```sh
uv venv -p 3.13 ~/venv_submit
. ~/venv_submit/bin/activate
uv pip install fluiddyn fluidsim ipython
```

Submit with

```sh
cd ~/dev/fluidsim/doc/examples/clusters/gricad_guix
. ~/venv_submit/bin/activate
./submit_bench_fluidsim.py
```
