import unittest
from copy import deepcopy

import numpy as np

from .solver import Simul

from fluiddyn.util import mpi

from fluidsim.util.testing import TestSimul, skip_if_no_fluidfft


skip_if_mpi = unittest.skipIf(
    mpi.nb_proc > 1, "MPI not implemented, for eg. sim.oper.gather_Xspace"
)


@skip_if_no_fluidfft
@skip_if_mpi
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
        params.output.periods_save.phys_fields = 0.2
        params.output.periods_plot.phys_fields = 0.0
        params.output.phys_fields.field_to_plot = "s"

    def test_simul(self):
        sim = self.sim
        sim.time_stepping.start()
        sim.plot_freq_diss()


@skip_if_no_fluidfft
@skip_if_mpi
class TestTimeStepping(TestSimul):
    Simul = Simul
    deltat = 2e-4
    k_init = 10
    amplitude = 0.7

    @classmethod
    def init_params(cls):

        params = cls.params = cls.Simul.create_default_params()

        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"

        params.oper.nx = 32
        params.oper.Lx = 2 * np.pi
        params.oper.coef_dealiasing = 0.66

        params.time_stepping.USE_CFL = False
        params.time_stepping.USE_T_END = False
        params.time_stepping.it_end = 0
        params.time_stepping.deltat0 = cls.deltat

        params.init_fields.type = "in_script"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        sim = cls.sim

        params2 = deepcopy(sim.params)
        params2.oper.nx = 64
        oper2 = type(sim.oper)(params2)

        cls.s_init = 1 + cls.amplitude * np.cos(cls.k_init * sim.oper.x)

        s_init2 = 1 + cls.amplitude * np.cos(cls.k_init * oper2.x)

        s_exact2 = s_init2 / (1 + cls.deltat * s_init2)
        s_exact2_fft = oper2.fft(s_exact2)
        cls.s_exact2_fft = s_exact2_fft[oper2.kx <= sim.oper.kx.max()]

    @classmethod
    def tearDownClass(cls):
        cls.sim.time_stepping.finalize_main_loop()

    def _test_type_time_scheme(
        self,
        type_time_scheme,
        coef_dealiasing=0.66,
        nb_pairs=1,
        nb_steps_compute_new_pair=None,
    ):
        sim = self.sim
        params = sim.params

        params2 = deepcopy(params)
        params2.oper.coef_dealiasing = coef_dealiasing
        sim.oper = type(sim.oper)(params2)

        sim.state.init_statephys_from(s=self.s_init.copy())
        sim.state.statespect_from_statephys()

        params.time_stepping.it_end += 1
        params.time_stepping.type_time_scheme = type_time_scheme
        params.time_stepping.phaseshift_random.nb_pairs = nb_pairs
        params.time_stepping.phaseshift_random.nb_steps_compute_new_pair = (
            nb_steps_compute_new_pair
        )
        sim.time_stepping.init_from_params()
        sim.time_stepping.main_loop()

        s_fft = sim.state.get_var("s_fft")
        assert np.allclose(s_fft, self.s_exact2_fft), abs(
            s_fft - self.s_exact2_fft
        ).max()

    def test_Euler(self):
        self._test_type_time_scheme("Euler")

    def test_Euler_phaseshift(self):
        self._test_type_time_scheme("Euler_phaseshift", 1)

    def test_Euler_phaseshift_random(self):
        self._test_type_time_scheme("Euler_phaseshift_random", 1)

    def test_Euler_phaseshift_random_bis(self):
        self._test_type_time_scheme("Euler_phaseshift_random", 1, 1, 1)

    def test_Euler_phaseshift_random_ter(self):
        self._test_type_time_scheme("Euler_phaseshift_random", 1, 2)

    def test_RK2(self):
        self._test_type_time_scheme("RK2")

    def test_RK2_trapezoid(self):
        self._test_type_time_scheme("RK2_trapezoid")

    def test_RK2_phaseshift(self):
        self._test_type_time_scheme("RK2_phaseshift", 1)

    def test_RK2_phaseshift_random(self):
        self._test_type_time_scheme("RK2_phaseshift_random", 1)

    def test_RK2_phaseshift_exact(self):
        self._test_type_time_scheme("RK2_phaseshift_exact", 1)

    def test_RK4(self):
        self._test_type_time_scheme("RK4")
