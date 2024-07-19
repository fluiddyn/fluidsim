(specifications->manifest
  (list "python-fluidfft"
	"coreutils"
    "python"
	"openmpi"
	"python-fluidfft-builder"
	"python-fluidfft-fftw"
	"python-fluidfft-fftwmpi"
	"python-fluidfft-mpi-with-fftw"
	"python-fluidfft-p3dfft"
	"python-fluidfft-pfft"
	"python-pytest"
	"python-pytest-allclose"
	"python-pytest-mock"
	"python-fluidsim"
    ; build dependencies for editable build
    "meson-python"
    "python-pythran"
    ; convenient to be able to check
    "which"
	"emacs"
  )
)
