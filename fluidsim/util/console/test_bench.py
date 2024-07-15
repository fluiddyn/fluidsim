"""Test benchmarking (:mod:`fluidsim.util.console.test_bench`)
==============================================================

"""

import sys

from fluiddyn.util import mpi
from fluidsim.util.testing import skip_if_no_fluidfft

from fluidsim.util.console.__main__ import run_bench, run_bench_analysis


@skip_if_no_fluidfft
def test2d(tmp_path):
    """Test launching ns2d benchmarks and plotting results."""

    if mpi.nb_proc > 1:
        type_fft = "fft2d.mpi_with_fftw1d"
    else:
        type_fft = "fft2d.with_pyfftw"

    command = f"fluidsim-bench 24 -d 2 -o {tmp_path} -t {type_fft}"
    sys.argv = command.split()
    run_bench()

    # Can plot only parallel benchmarks
    if mpi.rank == 0 and mpi.nb_proc != 1:
        command = f"fluidsim-bench-analysis 24 -d 2 -i {tmp_path}"
        sys.argv = command.split()
        run_bench_analysis()

    sys.argv = "fluidsim-bench -l -d 2".split()
    run_bench()

    if mpi.nb_proc == 1:
        return

    type_fft = "fft2d.mpi_with_fftw1d"

    sys.argv = f"fluidsim-bench -p -d 2 -t {type_fft}".split()
    run_bench()
    sys.argv = f"fluidsim-bench -e -d 2 -t {type_fft}".split()
    run_bench()


@skip_if_no_fluidfft
def test3d(tmp_path):
    """Test launching ns3d benchmarks and plotting results."""

    if mpi.nb_proc > 1:
        type_fft = "fft3d.mpi_with_fftw1d"
    else:
        type_fft = "fft3d.with_pyfftw"

    command = f"fluidsim-bench 8 -d 3 -o {tmp_path} -t {type_fft}"
    sys.argv = command.split()
    run_bench()

    sys.argv = "fluidsim-bench -l -d 3".split()
    run_bench()

    if mpi.nb_proc == 1:
        return

    type_fft = "fft3d.mpi_with_fftw1d"

    sys.argv = f"fluidsim-bench -p -d 3 -t {type_fft}".split()
    run_bench()
    sys.argv = f"fluidsim-bench -e -d 3 -t {type_fft}".split()
    run_bench()
