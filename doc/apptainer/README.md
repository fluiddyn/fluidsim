# Install Fluidsim from source in a Apptainer container

## Build the image locally on a node

```sh
apptainer build image-fluidsim.sif image-fluidsim.def
```

## Run the image to install Fluidsim from source

From the host:

```sh
mkdir -p ~/apptainer_dir/home
apptainer shell --no-home image-fluidsim.sif
```

From the container:

```sh
export HOME=$HOME/apptainer_dir/home
```
