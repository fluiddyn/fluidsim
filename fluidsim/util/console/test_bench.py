"""Test benchmarking (:mod:`fluidsim.util.console.test_bench`)
==============================================================

"""
import unittest
from shutil import rmtree
import sys

from fluiddyn.util import mpi
from fluidsim.util.testing import TestCase, skip_if_no_fluidfft

from fluidsim.util.console.__main__ import run_bench, run_bench_analysis


path_tmp = "/tmp/fluidsim_test_bench"


@skip_if_no_fluidfft
class TestBench(TestCase):
    """Test benchmarking."""

    @classmethod
    def setUpClass(cls):
        if mpi.rank == 0:
            rmtree(path_tmp, ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        if mpi.rank == 0:
            rmtree(path_tmp, ignore_errors=True)

    def test2d(self):
        """Test launching ns2d benchmarks and plotting results."""

        if mpi.nb_proc > 1:
            type_fft = "fft2d.mpi_with_fftw1d"
        else:
            type_fft = "fft2d.with_pyfftw"

        command = f"fluidsim-bench 24 -d 2 -o {path_tmp} -t {type_fft}"
        sys.argv = command.split()
        run_bench()

        # Can plot only parallel benchmarks
        if mpi.rank == 0 and mpi.nb_proc != 1:

            command = f"fluidsim-bench-analysis 24 -d 2 -i {path_tmp}"
            sys.argv = command.split()
            run_bench_analysis()

        sys.argv = "fluidsim-bench -l -d 2".split()
        run_bench()

        type_fft = "fft2d.mpi_with_fftw1d"

        sys.argv = f"fluidsim-bench -p -d 2 -t {type_fft}".split()
        run_bench()
        sys.argv = f"fluidsim-bench -e -d 2 -t {type_fft}".split()
        run_bench()

    def test3d(self):
        """Test launching ns3d benchmarks and plotting results."""

        if mpi.nb_proc > 1:
            type_fft = "fft3d.mpi_with_fftw1d"
        else:
            type_fft = "fft3d.with_pyfftw"

        command = f"fluidsim-bench 8 -d 3 -o {path_tmp} -t {type_fft}"
        sys.argv = command.split()
        run_bench()

        sys.argv = "fluidsim-bench -l -d 3".split()
        run_bench()

        type_fft = "fft3d.mpi_with_fftw1d"

        sys.argv = f"fluidsim-bench -p -d 3 -t {type_fft}".split()
        run_bench()
        sys.argv = f"fluidsim-bench -e -d 3 -t {type_fft}".split()
        run_bench()


if __name__ == "__main__":
    unittest.main()
