# Generate a PDF from a .md notebook with Myst

You can use the environment in ../../../../pixi-envs/env-myst and ~/dev/fluidsim/.venv
(read the README) launching:

```sh
pixi shell -m ../../../../pixi-envs/env-myst
. ~/dev/fluidsim/.venv/bin/activate
```

From this directory, a PDF can be produced with something like

```sh
# With default PATH_SIMUL and OUTPUT in the Makefile
make figures.pdf

# With desired PATH_SIMUL and OUTPUT file name
make figures.pdf PATH_SIMUL="/fsnet/project/meige/..." OUTPUT="figures_..._.pdf"
```

Of course, change the value of `PATH_SIMUL` and of the output file name!
