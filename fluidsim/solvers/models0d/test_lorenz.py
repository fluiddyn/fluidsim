import unittest

import fluiddyn.util.mpi as mpi

from .lorenz.solver import Simul

from fluidsim.util.testing import TestSimul


@unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
class TestLorenz(TestSimul):
    Simul = Simul

    @classmethod
    def init_params(cls):
        params = cls.params = cls.Simul.create_default_params()
        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"

        params.time_stepping.deltat0 = 0.02
        params.time_stepping.t_end = 0.04
        params.output.periods_print.print_stdout = 0.01

    def test_lorenz(self, params=None):
        """Should be able to run a base experiment."""

        sim = self.sim

        sim.state.state_phys.set_var("X", sim.Xs0 + 2.0)
        sim.state.state_phys.set_var("Y", sim.Ys0)
        sim.state.state_phys.set_var("Z", sim.Zs0)

        sim.time_stepping.start()

        sim.output.print_stdout.plot_deltat()
        sim.output.print_stdout.plot_XYZ()
        sim.output.print_stdout.plot_XZ()
        sim.output.print_stdout.plot_XY()
        sim.output.print_stdout.plot_XY_vs_time()


if __name__ == "__main__":
    unittest.main()
