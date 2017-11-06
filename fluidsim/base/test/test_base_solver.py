

import unittest
import shutil

import fluiddyn as fld
from fluiddyn.io import stdout_redirected
import fluiddyn.util.mpi as mpi
# to get fld.show
import fluiddyn.output

from fluidsim.base.solvers.base import SimulBase as Simul


class TestBaseSolver(unittest.TestCase):
    def setUp(self):
        params = Simul.create_default_params()

        params.short_name_type_run = 'test_base_solver'
        params.time_stepping.USE_CFL = False
        params.time_stepping.USE_T_END = False
        params.time_stepping.it_end = 4
        params.time_stepping.deltat0 = 0.1

        with stdout_redirected():
            self.sim = Simul(params)

    def tearDown(self):
        # clean by removing the directory
        if mpi.rank == 0:
            if hasattr(self, 'sim'):
                shutil.rmtree(self.sim.output.path_run)

    def test_simul(self):
        """Should be able to run a base experiment."""
        with stdout_redirected():
            self.sim.time_stepping.start()

        fld.show()


if __name__ == '__main__':
    unittest.main()
