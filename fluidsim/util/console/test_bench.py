"""Test benchmarking (:mod:`fluidsim.util.console.test_bench`)
==============================================================

"""
import unittest

from fluiddyn.io import stdout_redirected

from fluiddyn.util import mpi

from fluidsim.util.console.bench import bench, import_module_solver_from_key
from fluidsim.util.console.bench_analysis import plot_scaling


path_tmp = '/tmp/fluidsim_test_bench'


class TestBench(unittest.TestCase):
    """Test benchmarking."""
    n0 = 24

    def test2d(self):
        """Test launching ns2d benchmarks and plotting results."""
        n0 = self.n0
        with stdout_redirected():
            solver = import_module_solver_from_key('ns2d')
            bench(solver, dim='2d', n0=2*n0, n1=n0,
                  n2=None, path_dir=path_tmp, raise_error=True)

            # Can plot only parallel benchmarks
            if mpi.rank == 0 and mpi.nb_proc == 0:
                plot_scaling(
                    path_tmp, 'ns2d', 'any', 2 * n0, n0, show=False,
                    type_plot='weak')

    def test3d(self):
        """Test launching ns3d benchmarks and plotting results."""

        if mpi.nb_proc > 1:
            type_fft = 'fft3d.mpi_with_fftw1d'
        else:
            type_fft = 'fft3d.with_pyfftw'

        with stdout_redirected():
            solver = import_module_solver_from_key('ns3d')
            bench(
                solver, dim='3d', n0=8, n1=None, n2=None, path_dir=path_tmp,
                type_fft=type_fft, raise_error=True)


if __name__ == '__main__':
    unittest.main()
