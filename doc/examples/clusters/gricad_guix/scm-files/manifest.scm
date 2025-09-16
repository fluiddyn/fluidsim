(specifications->manifest
  (list "python-fluidfft"
    "coreutils"
    "guix"
    "python-wrapper@3.11.11"
    "openmpi@4.1.6"
    "python-mpi4py@3.1.4"
    "python-h5py-mpi"
    "python-fluidfft-builder"
    "python-fluidfft-fftw"
    "python-fluidfft-fftwmpi"
    "python-fluidfft-mpi-with-fftw"
    "python-fluidfft-p3dfft"
    "python-fluidfft-pfft"
    "python-pytest"
    "python-pytest-allclose"
    "python-pytest-mock"
    ; build dependencies for editable build
    "meson-python"
    "python-pythran"
    ; convenient to be able to check
    "which"
  )
)
