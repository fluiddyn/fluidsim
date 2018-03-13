

import unittest
import shutil

import numpy as np

import matplotlib
matplotlib.use('Agg')

import fluiddyn as fld
from fluiddyn.io import stdout_redirected
import fluiddyn.util.mpi as mpi

# to get fld.show
import fluiddyn.output

from fluidsim.base.solvers.pseudo_spect import SimulBasePseudoSpectral


class TestBaseSolverPS(unittest.TestCase):
    def setUp(self, params=None):
        """Should be able to run a base experiment."""

        if params is None:
            params = SimulBasePseudoSpectral.create_default_params()
            params.output.periods_plot.phys_fields = 0.
            params.output.periods_print.print_stdout = 0.
            params.short_name_type_run = 'test_base_solver_ps'

        nh = 8
        Lh = 2*np.pi
        params.oper.nx = nh
        params.oper.ny = nh
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.nu_2 = 1.

        params.time_stepping.t_end = 0.4

        with stdout_redirected():
            self.sim = SimulBasePseudoSpectral(params)

    def tearDown(self):
        # clean by removing the directory
        if mpi.rank == 0:
            if hasattr(self, 'sim'):
                self.sim.output.print_stdout.close()
                shutil.rmtree(self.sim.output.path_run)

    def test_simul(self):
        """Should be able to run a base experiment."""
        with stdout_redirected():
            self.sim.time_stepping.start()

        fld.show()


class TestOutputPS(TestBaseSolverPS):
    """Test a simulation run with online plotting and stdout printing."""
    def setUp(self):
        params = SimulBasePseudoSpectral.create_default_params()
        params.output.periods_plot.phys_fields = 0.2
        params.output.periods_print.print_stdout = 0.2
        params.short_name_type_run = 'test_output_ps'
        TestBaseSolverPS.setUp(self, params)


if __name__ == '__main__':
    unittest.main()
