

import unittest
import shutil

import numpy as np

import fluiddyn as fld
from fluiddyn.io import stdout_redirected
import fluiddyn.util.mpi as mpi

# to get fld.show
import fluiddyn.output

from fluidsim.base.solvers.pseudo_spect import SimulBasePseudoSpectral


class TestBaseSolverPS(unittest.TestCase):
    def setUp(self):
        """Should be able to run a base experiment."""

        params = SimulBasePseudoSpectral.create_default_params()

        params.short_name_type_run = 'test_base_solver_ps'

        nh = 16
        Lh = 2*np.pi
        params.oper.nx = nh
        params.oper.ny = nh
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.nu_2 = 1.

        params.time_stepping.t_end = 2.

        params.output.periods_plot.phys_fields = 0.
        params.output.periods_print.print_stdout = 0.

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


if __name__ == '__main__':
    unittest.main()
