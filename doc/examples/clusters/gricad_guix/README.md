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

## Current or pinned building versions

From now, there are two ways to build fluidsim on gricad:

- The first on is by following the four next sections (i.e. Sections
  [Setup Guix](#setup-guix) and
  [Install Fluidsim from source](#install-fluidsim-from-source), with the latter
  containing sub-sections
  [Change the changeset used for the Guix environment](#change-the-changeset-used-for-the-guix-environment)
  and [Build-install from source](#build-install-from-source)) in order to pull the
  current versions of guix with the current channel of gricad-guix-packages.

- The second one is by following Section
  [Pull pinned version of guix](#pull-pinned-version-of-guix)

## Setup Guix

The first thing to do, is to copy the file
`~/dev/fluidsim/doc/examples/clusters/gricad_guix/scm-files/channels.scm` into
`~/.config/guix/` or simply create it in `~/.config/guix/` with the following content:

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
terminal (replace `<changeset_ref>` with the chosen fluidsim changeset reference):

```sh
source /applis/site/guix-start.sh
hg clone https://foss.heptapod.net/fluiddyn/fluidsim ~/dev/fluidsim-clean
cd ~/dev/fluidsim-clean
hg up <changeset_ref> --clean
hg purge --all
guix hash -x -r .
```

Change the fluidsim changeset reference and the guix hash in
`~/dev/fluidsim/doc/examples/clusters/gricad_guix/scm-files/python-fluidsim.scm`
respectively at lines `(changeset "<changeset_ref>")))` and
`(base32 "<guix_hash_reference>"))))` that both appears twice in the file.

### Build-install from source

```sh
source /applis/site/guix-start.sh
DIR_MANIFEST=$HOME/dev/fluidsim/doc/examples/clusters/gricad_guix/scm-files
# This will take a while
guix package -f $DIR_MANIFEST/python-fluidsim.scm --manifest=$DIR_MANIFEST/manifest.scm --profile=$HOME/guix-profile-fluidsim
```

## Pull pinned version of guix

In the case of repeated errors while trying to follow the four previous sections (i.e.
Sections [Setup Guix](#setup-guix) and
[Install Fluidsim from source](#install-fluidsim-from-source), with the latter containing
sub-sections
[Change the changeset used for the Guix environment](#change-the-changeset-used-for-the-guix-environment)
and [Build-install from source](#build-install-from-source)) or simply in order to use a
stable process to build fluidsim environment on dahu, it is possible to build the
fluidsim environment from a pinned version of `guix` and `gricad-guix-packages`.

First, if you have not cloned fluidsim on dahu yet, follow the intro of Section
[Install Fluidsim from source](#install-fluidsim-from-source) (do not do the
sub-sections).

Then, open the file
`~/dev/fluidsim/doc/examples/clusters/gricad_guix/scm-files/python-fluidsim.scm`, specify
the fluidsim desired changeset reference (use hg lg to choose one and hg up to update
fluidsim to that one) at line `(changeset "<changeset_ref>")))` that appears twice in the
file. Specify the following pinned guix hash
`15sm4mknfagx1l4zgz49c2bfjjng8ykiz7jb45qa83jh03vzqc6a` at line
`(base32 "<guix_hash_reference>"))))` that also appears twice in the file.

Finally, launch the following command:

```sh
source /applis/site/guix-start.sh
DIR_MANIFEST=$HOME/dev/fluidsim/doc/examples/clusters/gricad_guix/scm-files
# This will take a while
guix time-machine -C $DIR_MANIFEST/channels-pinned.scm -- package -m $DIR_MANIFEST/manifest.scm -f $DIR_MANIFEST/python-fluidsim-pinned.scm --profile=$HOME/guix-profile-fluidsim
```

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
