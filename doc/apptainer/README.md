# Install Fluidsim from source in an Apptainer container

The directory `fluidsim/doc/apptainer` contains what is needed to run Fluidsim with Apptainer.

## Build the image locally on a node

From the host:

```sh
mkdir ~/apptainer
cd ~/apptainer
wget https://foss.heptapod.net/fluiddyn/fluidsim/-/raw/topic/default/apptainer/doc/apptainer/image-fluidsim.def
apptainer build image-fluidsim.sif image-fluidsim.def
```

## Run the image to install Fluidsim from source

From the host:

```sh
cd ~/apptainer
apptainer shell --no-home image-fluidsim.sif
```

From the container:

```sh
...
```
