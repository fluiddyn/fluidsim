# Using Fluidsim on Gricad clusters

We show in this directory
(<https://foss.heptapod.net/fluiddyn/fluidsim/-/tree/branch/default/doc/examples/clusters/gricad_guix>)
how to use Fluidsim on Gricad clusters. The main documentation for this HPC platform is
[here](https://gricad-doc.univ-grenoble-alpes.fr/hpc/). We will use
[Guix](https://gricad-doc.univ-grenoble-alpes.fr/hpc/softenv/guix/), which is one of the
recommended package managers for this platform.

## Get a login and setup ssh

Get an account on <https://perseus.univ-grenoble-alpes.fr/>.

Set an ssh key by following <https://gricad-doc.univ-grenoble-alpes.fr/hpc/connexion/>
and the alias

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

## Clone Fluidsim

Clone the Fluidsim repository in `$HOME/dev`:

```sh
hg clone https://foss.heptapod.net/fluiddyn/fluidsim ~/dev/fluidsim
```

Update to

```sh
cd ~/dev/fluidsim
hg up default
```

## Prepare the guix environment

In order to build fluidsim on dahu, one needs to use `guix` and [gricad-guix-package].
First, define the main directories used:

```sh
DIR_GRICAD_GUIX=$HOME/dev/fluidsim/doc/examples/clusters/gricad_guix
DIR_SCM_FILES=$DIR_GRICAD_GUIX/scm-files
```

You can access to the latter `cd $DIR_SCM_FILES` and check all the files needed to
prepare properly this environment:

- `$DIR_SCM_FILES/channels.scm`: gives the definition of the default channel used by
  `gricad-guix-packages` in the branch master for pulling the current version of `guix`.

- `$DIR_SCM_FILES/channels-pinned.scm`: gives a pinned (fixed in time) version of
  `gricad-guix-packages` and `guix`.

- `$DIR_SCM_FILES/manifest.scm`: gives the packages list needed to build Fluidsim with
  `guix`.

- `$DIR_SCM_FILES/python-fluidsim.scm`: gives the exact version of Fluidsim to be build
  by `guix`.

There are then two ways of building Fluidsim: pull a pinned or the current version of
`guix` and `gricad-guix-packages`.

### Pull pinned version of guix

In order to use a stable process to build fluidsim environment on dahu, it is possible to
build the fluidsim environment from a pinned version of `guix` and `gricad-guix-packages`
by launching the following command:

```sh
source /applis/site/guix-start.sh
# This will take a while
guix time-machine -C $DIR_SCM_FILES/channels-pinned.scm -- \
  package -m $DIR_SCM_FILES/manifest.scm -f $DIR_SCM_FILES/python-fluidsim.scm \
  --profile=$HOME/guix-profile-fluidsim
```

### Use current Guix and gricad-guix-packages version

#### Setup Guix

```sh
cp $DIR_SCM_FILES/channels.scm ~/.config/guix/
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

#### Build-install from source

```sh
source /applis/site/guix-start.sh
# This will take a while
guix package -f $DIR_SCM_FILES/python-fluidsim.scm --manifest=$DIR_SCM_FILES/manifest.scm --profile=$HOME/guix-profile-fluidsim
```

### Change scm-files/python-fluidsim.scm (exact Fluidsim version)

If you want to choose a given version of Fluidsim that is not from the default one,
follow this section. One needs to choose a changeset (a commit) and get its changeset
reference. One can study them with:

```sh
cd ~/dev/fluidsim
# get the node (changeset reference) of the current commit
# (you can choose this commit)
hg log -r . -T "{node}"
# study all commits
# (you can choose another commit)
hg log -G
```

Get the "Guix hash" (the guix hash is a reference sequence related to a given version of
`guix` with Fluidsim) with:

```sh
source /applis/site/guix-start.sh
hg clone https://foss.heptapod.net/fluiddyn/fluidsim ~/dev/fluidsim-clean
cd ~/dev/fluidsim-clean
hg up <changeset_ref> --clean
hg purge --all
guix hash -x -r .
```

Replace `<changeset_ref>` with the chosen fluidsim changeset reference.

```{note}
`guix download` does not support Mercurial.
```

Change the fluidsim changeset reference and the guix hash in
`$DIR_SCM_FILES/python-fluidsim.scm` respectively at lines
`(changeset "<changeset_ref>")))` and `(base32 "<guix_hash_reference>"))))` that both
appears twice in the file.

## List the packages

Once the fluidsim profile is created, it can be useful to list the package installed in
the profile by:

```sh
guix package --list-installed --profile=$HOME/guix-profile-fluidsim
```

## Source the environment

Now that the profile is created, in order to use the `guix-profile-fluidsim` environment,
you need to source it with the following command:

```sh
source $HOME/guix-profile-fluidsim/etc/profile
```

## Test Fluidsim in sequential

```sh
python -m pytest --pyargs fluidsim
```

## Submit a Fluidfft benchmark

```sh
ssh dahu-oar3
cd $DIR_GRICAD_GUIX
source $HOME/guix-profile-fluidsim/etc/profile
oarsub -S ./job_fluidfft_bench.oar
```

## Submit a Fluidsim benchmark

Here, we are going to show how to do it with two strategies, either manually write a OAR
script or use fluiddyn to write it.

### Hand written OAR script

```sh
ssh dahu-oar3
cd $DIR_GRICAD_GUIX
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
cd $DIR_GRICAD_GUIX
. ~/venv_submit/bin/activate
./submit_bench_fluidsim.py
```

[gricad-guix-package]: https://gricad-gitlab.univ-grenoble-alpes.fr/bouttiep/gricad_guix_packages
