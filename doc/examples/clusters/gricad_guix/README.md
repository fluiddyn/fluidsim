# Using Fluidsim on Gricad clusters

We show in this directory
(<https://foss.heptapod.net/fluiddyn/fluidsim/-/tree/branch/default/doc/examples/clusters/gricad_guix>)
how to use Fluidsim on Gricad clusters. The main documentation for this HPC platform is
[here](https://gricad-doc.univ-grenoble-alpes.fr/hpc/). We will use
[Guix](https://gricad-doc.univ-grenoble-alpes.fr/hpc/softenv/guix/), which is one of the
recommended package managers for this platform.

## Get a login and setup ssh

Get an account on <https://perseus.univ-grenoble-alpes.fr/> and setup everything to
connect on Dahu by following <https://gricad-doc.univ-grenoble-alpes.fr/hpc/connexion/>.
It is also very convenient to create an alias by adding in your `~/.bashrc`:

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

## Clone the Fluidsim repository

Clone the Fluidsim repository in `~/dev`:

```sh
hg clone https://foss.heptapod.net/fluiddyn/fluidsim ~/dev/fluidsim
```

Later on, one can update with

```sh
cd ~/dev/fluidsim
hg pull
hg up default
```

## Prepare the Guix environment

We want to create a Guix environment for Fluidsim simulations. We need to use
[gricad-guix-packages], an alternative Guix channel maintained by Gricad people.
Therefore, our environment depends on

- the exact Guix version,
- the exact [gricad-guix-packages] version,
- the exact Fluidsim version.

We first define variables containing paths towards directories containing files defining
our Guix environment:

```sh
DIR_GRICAD_GUIX=$HOME/dev/fluidsim/doc/examples/clusters/gricad_guix
DIR_SCM_FILES=$DIR_GRICAD_GUIX/scm-files
```

The files needed to prepare the possible environments are:

- `$DIR_SCM_FILES/channels.scm`: file to be used as `~/.config/guix/channels.scm` so that
  Guix uses the master branch of [gricad-guix-packages].

- `$DIR_SCM_FILES/channels-pinned.scm`: specifies alternative channels with pinned (fixed
  in time) versions of [gricad-guix-packages] and Guix.

- `$DIR_SCM_FILES/manifest.scm`: contains the list of packages in the environment.

- `$DIR_SCM_FILES/python-fluidsim.scm`: redefinition of the Fluidsim Guix package with an
  unreleased version of Fluidsim.

One can use these different files as needed in different combinaisons. We present here
two possibilities.

### Pull pinned version of Guix and gricad-guix-packages

In order to use a stable process to build Fluidsim environment on Dahu, it is possible to
build the environment from a pinned version of Guix and [gricad-guix-packages]:

```sh
source /applis/site/guix-start.sh
# This will take a while
guix time-machine -C $DIR_SCM_FILES/channels-pinned.scm -- package \
  -m $DIR_SCM_FILES/manifest.scm -f $DIR_SCM_FILES/python-fluidsim.scm \
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

You only need to update Guix (and thus run `guix pull`) when a package you want to use
has been created or updated.

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
guix package -f $DIR_SCM_FILES/python-fluidsim.scm --manifest=$DIR_SCM_FILES/manifest.scm \
  --profile=$HOME/guix-profile-fluidsim
```

```{note}
Without `-f $DIR_SCM_FILES/python-fluidsim.scm`, the Fluidsim version
taken from [gricad-guix-packages] would be used.
```

### Update scm-files/python-fluidsim.scm to use another Fluidsim version

If you want to choose a given version of Fluidsim that is not from the default one,
follow this section. One needs to choose a changeset (a commit) and get its changeset
identifier. One can study them with:

```sh
cd ~/dev/fluidsim
# get the node (changeset identifier) of the current commit
# (you can choose this commit)
hg log -r . -T "{node}"
# study all commits
# (you can choose another commit)
hg log -G
```

We now need to get the "Guix hash" corresponding to the specific version of Fluidsim that
we want to use. This is a hash (an long hexadecimal number) computed by Guix from the
source of Fluidsim. It is used at build time by Guix for security to check if what we get
indeed corresponds to the specified version. Getting the Guix hash of something is
usually done with the command `guix download`, but unfortunately, it does not yet support
Mercurial, so one needs to run:

```sh
source /applis/site/guix-start.sh
hg clone https://foss.heptapod.net/fluiddyn/fluidsim ~/dev/fluidsim-clean
cd ~/dev/fluidsim-clean
hg up <changeset_id> --clean
hg purge --all
guix hash -x -r .
```

Of course, `<changeset_id>` has to be replaced with the chosen Fluidsim changeset
identifier.

Change the Fluidsim changeset identifier and the guix hash in
`$DIR_SCM_FILES/python-fluidsim.scm` respectively at lines
`(changeset "<changeset_id>")))` and `(base32 "<guix_hash_identifier>"))))` that both
appears twice in the file.

## List the packages

Once the Fluidsim profile is created, it can be useful to list the package installed in
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
script or use Fluiddyn to write it.

### Hand written OAR script

```sh
ssh dahu-oar3
cd $DIR_GRICAD_GUIX
source $HOME/guix-profile-fluidsim/etc/profile
oarsub -S ./job_fluidsim_bench.oar
```

### With Fluiddyn

Prepare a virtual env (1 time). From a new terminal:

```sh
uv venv -p 3.13 ~/venv_submit
. ~/venv_submit/bin/activate
# note: this environment is NOT going to be used during the simulation
uv pip install fluiddyn fluidsim ipython
```

Submit with

```sh
. ~/venv_submit/bin/activate
cd $DIR_GRICAD_GUIX
./submit_bench_fluidsim.py
```

[gricad-guix-packages]: https://gricad-gitlab.univ-grenoble-alpes.fr/bouttiep/gricad_guix_packages
