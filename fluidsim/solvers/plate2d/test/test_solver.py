import unittest
import shutil


import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected

from fluidsim.solvers.plate2d.solver import Simul


class TestSolverPLATE2D(unittest.TestCase):
    # @unittest.expectedFailure
    def test_tendency(self):

        params = Simul.create_default_params()

        params.short_name_type_run = 'test'

        nh = 64
        params.oper.nx = nh
        params.oper.ny = nh
        Lh = 6.
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.oper.coef_dealiasing = 2./3
        params.nu_8 = 2.

        params.time_stepping.USE_CFL = False
        params.time_stepping.deltat0 = 0.005
        params.time_stepping.t_end = 0.5

        params.init_fields.type = 'noise'
        params.output.HAS_TO_SAVE = False
        params.FORCING = False

        params.output.ONLINE_PLOT_OK = False

        with stdout_redirected():
            sim = Simul(params)

        ratio = sim.test_tendencies_nonlin()

        self.assertGreater(1e-15, ratio)

        if mpi.rank == 0:
            shutil.rmtree(sim.output.path_run)


if __name__ == '__main__':
    unittest.main()
