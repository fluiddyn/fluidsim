import unittest
import warnings

try:
    import scipy.sparse

    scipy_installed = True
except ImportError:
    scipy_installed = False


from fluiddyn.util import mpi

from fluidsim.solvers.ad1d.solver import Simul

from fluidsim.util.testing import TestSimul


@unittest.skipIf(not scipy_installed, "No module named scipy.sparse")
class TestSolverAD1D(TestSimul):

    Simul = Simul

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # RuntimeWarnings are very common when numpy installed and numpy
        # used to build scipy don't match.
        # See:
        # [1] http://thread.gmane.org/gmane.comp.python.cython.devel/14352/focus=14354  # noqa
        # [2] https://stackoverflow.com/a/40846742
        warnings.filterwarnings(
            "ignore", "^numpy.ufunc size changed", RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore", "^numpy.dtype size changed", RuntimeWarning
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        warnings.resetwarnings()

    @classmethod
    def init_params(cls):
        params = cls.params = cls.Simul.create_default_params()

        params.U = 1.0

        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"

        params.oper.nx = 40
        params.oper.Lx = 1.0

        params.time_stepping.type_time_scheme = "RK2"

        params.nu_2 = 0.01

        params.time_stepping.t_end = 0.4
        params.time_stepping.USE_CFL = True

        params.init_fields.type = "gaussian"

        params.output.periods_print.print_stdout = 0.25
        params.output.periods_save.phys_fields = 0.5
        params.output.periods_plot.phys_fields = 0.0
        params.output.phys_fields.field_to_plot = "s"

    @unittest.skipIf(
        mpi.nb_proc > 1, "MPI not implemented, for eg. sim.oper.gather_Xspace"
    )
    def test_simul(self):
        sim = self.sim
        sim.time_stepping.start()

        sim.state.compute("dx_s")
        dx_s = sim.state.compute("dx_s")

        with self.assertRaises(ValueError):
            sim.state.compute("bar")

        if hasattr(sim.oper, "identity"):
            sim.oper.identity()
            sim.oper.pxx(dx_s)

        sim.output.phys_fields.plot()
        sim.output.phys_fields.plot(field="s", time=10)
        sim.output.phys_fields.animate()
        sim.output.phys_fields.movies.update_animation(1)


@unittest.skipIf(not scipy_installed, "No module named scipy.sparse")
class TestInitAD1D(TestSimul):
    Simul = Simul

    @classmethod
    def init_params(cls):
        params = cls.params = cls.Simul.create_default_params()
        params.output.HAS_TO_SAVE = False
        params.init_fields.type = "cos"

    @unittest.skipIf(
        mpi.nb_proc > 1, "MPI not implemented, for eg. sim.oper.gather_Xspace"
    )
    def test_init(self):
        """Only test the initialization"""


if __name__ == "__main__":
    unittest.main()
