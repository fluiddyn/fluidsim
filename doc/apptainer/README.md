# Install Fluidsim from source in an Apptainer container

The directory `fluidsim/doc/apptainer` contains what is needed to run Fluidsim with
Apptainer. See also https://gricad-doc.univ-grenoble-alpes.fr/hpc/softenv/container/.

## Build the image locally on a node

From the host:

```sh
mkdir ~/apptainer
cd ~/apptainer
wget https://foss.heptapod.net/fluiddyn/fluidsim/-/raw/topic/default/apptainer/doc/apptainer/image-fluidsim.def
apptainer build image-fluidsim.sif image-fluidsim.def
```

## Run the image to install Fluidsim from source

From the host (a node of the cluster!):

```sh
cd ~/apptainer
apptainer shell --no-home image-fluidsim.sif
```

From the container:

```sh
python -m venv venv-fluidsim
. venv-fluidsim/bin/activate
HDF5_MPI="ON" pip install --no-binary=h5py h5py pytest pytest-mpi
python -c 'import h5py; h5py.run_tests()'
hg clone https://foss.heptapod.net/fluiddyn/fluidsim
cd fluidsim/
pip install ".[test,mpi,fft]"
pytest --pyargs fluidsim
pip install fluidfft-fftw fluidfft-fftwmpi fluidfft-mpi_with_fftw
mpirun -np 2 pytest --pyargs fluidsim
pip install fluidfft-p3dfft
pip install fluidfft-pfft
```
