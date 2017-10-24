
import unittest
import shutil
import warnings

import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected

from fluidsim.solvers.ad1d.solver import Simul


class TestSolverAD1D(unittest.TestCase):

    def setUp(self):
        # RuntimeWarnings are very common when numpy installed and numpy
        # used to build scipy don't match.
        # See:
        # [1] http://thread.gmane.org/gmane.comp.python.cython.devel/14352/focus=14354  # noqa
        # [2] https://stackoverflow.com/a/40846742
        warnings.filterwarnings(
            "ignore", "^numpy.ufunc size changed", RuntimeWarning)
        warnings.filterwarnings(
            "ignore", "^numpy.dtype size changed", RuntimeWarning)
        params = Simul.create_default_params()

        params.U = 1.

        params.short_name_type_run = 'test'

        params.oper.nx = 40
        params.oper.Lx = 1.

        params.time_stepping.type_time_scheme = 'RK2'

        params.nu_2 = 0.01

        params.time_stepping.t_end = 0.4
        params.time_stepping.USE_CFL = True

        params.init_fields.type = 'gaussian'

        params.output.periods_print.print_stdout = 0.25

        params.output.periods_save.phys_fields = 0.5

        params.output.periods_plot.phys_fields = 0.

        params.output.phys_fields.field_to_plot = 's'

        with stdout_redirected():
            self.sim = Simul(params)

    def tearDown(self):
        # clean by removing the directory
        if mpi.rank == 0:
            if hasattr(self, 'sim'):
                shutil.rmtree(self.sim.output.path_run)

        warnings.resetwarnings()

    @unittest.skipIf(
        mpi.nb_proc > 1, 'MPI not implemented, for eg. sim.oper.gather_Xspace')
    def test_simul(self):
        with stdout_redirected():
            self.sim.time_stepping.start()


if __name__ == '__main__':
    unittest.main()
