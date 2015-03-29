import unittest
import shutil

# import numpy as np

import fluiddyn as fld

from fluiddyn.io import stdout_redirected


class TestSolverPLATE2D(unittest.TestCase):
    # @unittest.expectedFailure
    def test_tendency(self):

        key_solver = 'PLATE2D'
        solver = fld.simul.import_module_solver_from_key(key_solver)
        params = fld.simul.create_params(solver)

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

        params.init_fields.type_flow_init = 'NOISE'
        params.output.HAS_TO_SAVE = False
        params.FORCING = False

        params.output.ONLINE_PLOT_OK = False

        with stdout_redirected():
            sim = solver.Simul(params)

        ratio = sim.test_tendencies_nonlin()

        self.assertGreater(1e-15, ratio)

        shutil.rmtree(sim.output.path_run)


if __name__ == '__main__':
    unittest.main()
