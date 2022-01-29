import unittest

import numpy as np

import fluiddyn.util.mpi as mpi


from fluidsim.util.testing import TestSimul, skip_if_no_fluidfft, classproperty


@skip_if_no_fluidfft
class TestSimulBase(TestSimul):
    @classproperty
    def Simul(cls):
        from fluidsim.solvers.plate2d.solver import Simul

        return Simul

    @classmethod
    def init_params(cls, Lh=2, nh=32):
        cls.params = params = cls.Simul.create_default_params()
        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"
        cls.nh = nh
        params.oper.nx = nh
        params.oper.ny = nh
        cls.Lh = Lh
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.oper.coef_dealiasing = 2.0 / 3
        params.nu_8 = 2.0

        params.time_stepping.USE_CFL = False
        params.time_stepping.deltat0 = 0.005
        params.time_stepping.t_end = 0.5
        params.init_fields.type = "noise"
        params.output.HAS_TO_SAVE = False
        params.forcing.enable = False

        params.output.ONLINE_PLOT_OK = False
        return params


class TestSolverPlate2DTendency(TestSimulBase):
    def test_tendency(self):
        ratio = self.sim.test_tendencies_nonlin()
        self.assertGreater(2e-15, ratio)


@unittest.skipIf(mpi.nb_proc > 1, "plot function works sequentially only")
class TestSolverPlate2DInit(TestSimulBase):
    @classmethod
    def init_params(cls):
        params = super().init_params(4.0, nh=8)
        params.init_fields.type = "harmonic"
        params.init_fields.harmonic.i0 = 2
        params.init_fields.harmonic.i1 = 4

    def test_init(self):
        pass


class TestSolverPlate2DOutput(TestSimulBase):
    @classmethod
    def init_params(cls):
        params = super().init_params(6.0, nh=12)
        delta_x = cls.Lh / cls.nh

        kmax = np.sqrt(2) * np.pi / delta_x
        deltat = 2 * np.pi / kmax**2 / 2

        params.time_stepping.deltat0 = deltat
        params.time_stepping.it_end = 10
        params.time_stepping.USE_T_END = False

        params.init_fields.type = "noise"
        params.init_fields.noise.velo_max = 1e-6

        params.forcing.enable = True
        params.forcing.type = "tcrandom"
        params.forcing.forcing_rate = 1e4
        params.forcing.nkmax_forcing = 5
        params.forcing.nkmin_forcing = 2
        params.forcing.tcrandom.time_correlation = 100 * deltat

        params.nu_8 = (
            2e1 * params.forcing.forcing_rate ** (1.0 / 3) * delta_x**8
        )
        params.output.HAS_TO_SAVE = True
        params.output.periods_print.print_stdout = 2 * deltat

        params.output.periods_save.phys_fields = 6 * deltat
        params.output.periods_save.spectra = 6 * deltat
        params.output.periods_save.spatial_means = 4 * deltat
        params.output.periods_save.correl_freq = 2 * deltat

        params.output.ONLINE_PLOT_OK = True
        params.output.period_refresh_plots = 6 * deltat

        params.output.spatial_means.HAS_TO_PLOT_SAVED = True
        params.output.spectra.HAS_TO_PLOT_SAVED = True

        params.output.correl_freq.HAS_TO_PLOT_SAVED = True
        params.output.correl_freq.it_start = 0
        params.output.correl_freq.nb_times_compute = 5
        params.output.correl_freq.coef_decimate = 1
        params.output.correl_freq.iomegas1 = [1, 2]

    def test_output(self):

        sim = self.sim

        sim.time_stepping.start()

        sim.output.correl_freq.compute_corr4_norm()

        for key in sim.info_solver.classes.State.keys_computable:
            sim.state.compute(key)

        sim.state.compute(key)

        with self.assertRaises(ValueError):
            sim.state.compute("abcdef")

        sim.state.compute("abcdef", RAISE_ERROR=False)

        if mpi.nb_proc > 1:
            return

        sim.output.correl_freq.plot_norm_pick_corr4()
        sim.output.correl_freq.plot_convergence()
        sim.output.correl_freq.plot_corr2()
        sim.output.correl_freq.plot_corr2_1d()
        sim.output.correl_freq.plot_corr4()
        sim.output.correl_freq.plot_convergence2()

        sim.output.spatial_means.plot(with_dtE=True)

        sim.output.spectra.plot1d()
        sim.output.spectra.plot2d()


if __name__ == "__main__":
    unittest.main()
