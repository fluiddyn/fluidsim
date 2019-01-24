import unittest

import fluiddyn.util.mpi as mpi

from .predaprey.solver import Simul

from fluidsim.util.testing import TestSimul


@unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
class TestLorenz(TestSimul):
    Simul = Simul

    @classmethod
    def init_params(cls):
        params = cls.params = cls.Simul.create_default_params()
        params.time_stepping.deltat0 = 0.02
        params.time_stepping.t_end = 0.04
        params.output.periods_print.print_stdout = 0.01

    def test_predaprey(self, params=None):
        """Should be able to run a base experiment."""

        sim = self.sim

        sim.state.state_phys.set_var("X", sim.Xs + 2.0)
        sim.state.state_phys.set_var("Y", sim.Ys + 1.0)

        sim.time_stepping.start()

        sim.output.print_stdout.plot_XY()
        sim.output.print_stdout.plot_XY_vs_time()
        sim.output.print_stdout.plot_deltat()
        sim.output.print_stdout.plot_potential()


if __name__ == "__main__":
    unittest.main()
