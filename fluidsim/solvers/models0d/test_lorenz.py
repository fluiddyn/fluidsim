
import unittest
import shutil

import matplotlib
matplotlib.use('Agg')

from fluiddyn.io import stdout_redirected
import fluiddyn.util.mpi as mpi

from .lorenz.solver import Simul


class TestLorenz(unittest.TestCase):

    def tearDown(self):
        # clean by removing the directory
        if mpi.rank == 0:
            if hasattr(self, 'sim'):
                self.sim.output.print_stdout.close()
                shutil.rmtree(self.sim.output.path_run)

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_lorenz(self, params=None):
        """Should be able to run a base experiment."""

        params = Simul.create_default_params()
        params.time_stepping.deltat0 = 0.02
        params.time_stepping.t_end = 0.04

        params.output.periods_print.print_stdout = 0.01

        with stdout_redirected():
            sim = Simul(params)

        sim.state.state_phys.set_var('X', sim.Xs0 + 2.)
        sim.state.state_phys.set_var('Y', sim.Ys0)
        sim.state.state_phys.set_var('Z', sim.Zs0)

        with stdout_redirected():
            sim.time_stepping.start()

        sim.output.print_stdout.plot_XYZ()
        sim.output.print_stdout.plot_XZ()

if __name__ == '__main__':
    unittest.main()
