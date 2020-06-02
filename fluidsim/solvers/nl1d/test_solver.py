import unittest

from .solver import Simul

from fluiddyn.util import mpi

from fluidsim.util.testing import TestSimul


class TestSolverSquare1D(TestSimul):
    Simul = Simul

    @classmethod
    def init_params(cls):

        params = cls.params = cls.Simul.create_default_params()

        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"

        params.oper.nx = 40
        params.oper.Lx = 1.0

        params.nu_2 = 0.01

        params.time_stepping.type_time_scheme = "Euler"
        params.time_stepping.USE_CFL = False
        params.time_stepping.USE_T_END = False
        params.time_stepping.it_end = 4
        params.time_stepping.deltat_max = 0.1

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
        params = sim.params

        sim.time_stepping.main_loop(print_begin=True, save_init_field=True)

        params.time_stepping.it_end += 2
        params.time_stepping.type_time_scheme = "Euler_phaseshift"
        sim.time_stepping.init_from_params()
        sim.time_stepping.main_loop()

        params.time_stepping.it_end += 2
        params.time_stepping.type_time_scheme = "RK2_trapezoid"
        sim.time_stepping.init_from_params()
        sim.time_stepping.main_loop()

        params.time_stepping.it_end += 2
        params.time_stepping.type_time_scheme = "RK2_phaseshift"
        sim.time_stepping.init_from_params()
        sim.time_stepping.main_loop()

        sim.time_stepping.finalize_main_loop()

        sim.plot_freq_diss()
