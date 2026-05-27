# Generate a PDF from a .md notebook with Myst

You can use the environment in ../../../../pixi-envs/env-myst
(read the README).

From this directory, a PDF can be produced with something like

```sh
PATH_SIMUL_DIR="0" myst build --execute --typst notebook.md -o figures.pdf
```

Of course, change the value of `PATH_SIMUL_DIR` and of the output file!
