"""Test benchmarking (:mod:`fluidsim.util.console.test_bench`)
==============================================================

"""
import unittest

from fluiddyn.io import stdout_redirected

from fluiddyn.util import mpi

from fluidsim.util.console.bench import bench, import_module_solver_from_key
from fluidsim.util.console.bench_analysis import plot_scaling


path_tmp = '/tmp/fluidsim_test_bench'


class TestsBench(unittest.TestCase):
    """Test benchmarking."""

    def test2d(self):
        """Test launching ns2d benchmarks and plotting results."""
        n0 = 24
        with stdout_redirected():
            solver = import_module_solver_from_key('ns2d')
            bench(solver, dim='2d', n0=24, n1=None, n2=None, path_dir=path_tmp)
            if mpi.nb_proc > 1 and mpi.rank == 0:
                plot_scaling(path_tmp, 'ns2d', 'any', n0, n0, '2d', show=False)
                plot_scaling(
                    path_tmp, 'ns2d', 'any', n0, n0, '2d', show=False,
                    type_plot='weak')

    def test3d(self):
        """Test launching ns3d benchmarks and plotting results."""
        with stdout_redirected():
            solver = import_module_solver_from_key('ns3d')
            bench(solver, dim='3d', n0=8, n1=None, n2=None, path_dir=path_tmp)


if __name__ == '__main__':
    unittest.main()
