# Pixi environments for Fluidsim

With a local version of the repo:

```sh
cd ~/dev/fluidsim/pixi-envs/env-fluidsim
pixi shell
```

or

```sh
pixi shell --manifest-path ~/dev/fluidsim/pixi-envs/env-fluidsim
```

To record a kernel usable through Jupyter Lab:

```sh
pixi run --manifest-path ~/dev/fluidsim/pixi-envs/env-fluidsim python -m ipykernel install --user --name=env-fluidsim
```
