import unittest

import numpy as np
import matplotlib.pyplot as plt

import fluidsim as fls

import fluiddyn.util.mpi as mpi

from fluidsim.solvers.ns2d.solver import Simul
from fluidsim.util.testing import TestSimul


class TestSimulBase(TestSimul):
    Simul = Simul

    @classmethod
    def init_params(cls):

        params = cls.params = cls.Simul.create_default_params()
        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"

        nh = 32
        params.oper.nx = nh
        params.oper.ny = nh
        Lh = 6.0
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.oper.coef_dealiasing = 2.0 / 3
        params.nu_8 = 2.0

        params.time_stepping.t_end = 0.5

        params.init_fields.type = "noise"

        return params


class TestSolverNS2DTendency(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()

        params.output.HAS_TO_SAVE = False

    def test_tendency(self):
        sim = self.sim
        rot_fft = sim.state.get_var("rot_fft")

        tend = sim.tendencies_nonlin(state_spect=sim.state.state_spect)
        Frot_fft = tend.get_var("rot_fft")

        T_rot = np.real(Frot_fft.conj() * rot_fft)

        ratio = sim.oper.sum_wavenumbers(T_rot) / sim.oper.sum_wavenumbers(
            abs(T_rot)
        )

        self.assertGreater(1e-15, abs(ratio))


class TestForcingProportional(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.forcing.enable = True
        params.forcing.type = "proportional"

    def test_proportional(self):
        self.sim.time_stepping.start()
        self.sim.forcing.forcing_maker.verify_injection_rate()
        self.sim.forcing.forcing_maker.verify_injection_rate_coarse()


class TestForcing(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.forcing.enable = True
        params.forcing.type = "tcrandom"
        params.forcing.normalized.type = "particular_k"

    def test_(self):
        self.sim.time_stepping.start()


class TestForcingConstantRateEnergy(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.forcing.enable = True
        params.forcing.type = "tcrandom"
        params.forcing.normalized.constant_rate_of = "energy"
        params.forcing.forcing_rate = 3.333
        params.output.periods_save.spatial_means = 1e-6

    def test_(self):
        self.sim.time_stepping.start()

        if mpi.rank == 0:
            # Does the energy injection rate have the correct value at all times ?
            means = self.sim.output.spatial_means.load()
            assert np.allclose(
                means["PK_tot"], self.sim.params.forcing.forcing_rate
            )


class TestForcingOutput(TestSimulBase):
    @classmethod
    def init_params(self):

        params = super().init_params()
        params.forcing.enable = True
        params.forcing.type = "tcrandom"

        params.time_stepping.max_elapsed = "0:10:00"

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.2

        params.output.ONLINE_PLOT_OK = True
        params.output.periods_print.print_stdout = 0.2
        params.output.periods_plot.phys_fields = 0.2

        params.output.increments.HAS_TO_PLOT_SAVED = True

        for tag in params.output._tag_children:
            if tag.startswith("periods"):
                continue
            child = params.output[tag]
            if hasattr(child, "HAS_TO_PLOT_SAVED"):
                child["HAS_TO_PLOT_SAVED"] = True

    def test_forcing_output(self):

        sim = self.sim

        sim.time_stepping.start()

        sim.state.compute("rot_fft")
        sim.state.compute("rot_fft")

        with self.assertRaises(ValueError):
            sim.state.compute("abc")
        sim.state.compute("abc", RAISE_ERROR=False)

        ux_fft = sim.state.compute("ux_fft")
        uy_fft = sim.state.compute("uy_fft")

        sim.state.init_statespect_from(ux_fft=ux_fft)
        sim.state.init_statespect_from(uy_fft=uy_fft)

        if mpi.nb_proc == 1:
            sim.output.spectra.plot1d()
            sim.output.spectra.plot2d()
            sim.output.spectra.load2d_mean()
            sim.output.spectra.load1d_mean()

            sim.output.spatial_means.plot()
            sim.output.spatial_means.plot_dt_energy()
            sim.output.spatial_means.plot_dt_enstrophy()
            sim.output.spatial_means.compute_time_means()
            sim.output.spatial_means.load_dataset()
            sim.output.spatial_means.time_first_saved()
            sim.output.spatial_means.time_last_saved()

            sim.output.print_stdout.plot_energy()
            sim.output.print_stdout.plot_deltat()

            sim.output.spectra_multidim.plot()

            sim.output.spect_energy_budg.plot()
            with self.assertRaises(ValueError):
                sim.state.get_var("test")

            sim2 = fls.load_sim_for_plot(sim.output.path_run)
            sim2.output

            sim2.output.increments.load()
            sim2.output.increments.plot()
            sim2.output.increments.load_pdf_from_file()
            sim2.output.increments.plot_pdf()

            sim2.output.phys_fields.animate(
                "ux",
                dt_frame_in_sec=1e-6,
                dt_equations=0.3,
                repeat=False,
                clim=(-1, 1),
                save_file=False,
                numfig=1,
            )
            sim2.output.phys_fields.plot()

            # `compute('q')` two times for better coverage...
            sim.state.get_var("q")
            sim.state.get_var("q")
            sim.state.get_var("div")

            path_run = sim.output.path_run
            if mpi.nb_proc > 1:
                path_run = mpi.comm.bcast(path_run)

            sim3 = fls.load_state_phys_file(path_run, modif_save_params=False)
            sim3.params.time_stepping.t_end += 0.2
            sim3.time_stepping.start()

            if mpi.nb_proc == 1:
                sim3.output.phys_fields.animate(
                    "ux",
                    dt_frame_in_sec=1e-6,
                    dt_equations=0.3,
                    repeat=False,
                    clim=(-1, 1),
                    save_file=False,
                    numfig=1,
                )
        plt.close("all")

        # test_enstrophy_conservation
        # Verify that the enstrophy growth rate due to nonlinear tendencies
        # (advection term) must be zero.
        self.sim.params.forcing.enable = False
        tendencies_fft = self.sim.tendencies_nonlin()
        state_spect = self.sim.state.state_spect
        oper = self.sim.oper
        Frot_fft = tendencies_fft.get_var("rot_fft")
        rot_fft = state_spect.get_var("rot_fft")

        T_rot = (Frot_fft.conj() * rot_fft + Frot_fft * rot_fft.conj()).real / 2.0
        sum_T = oper.sum_wavenumbers(T_rot)
        self.assertAlmostEqual(sum_T, 0, places=14)


class TestSolverNS2DInitJet(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.init_fields.type = "jet"
        params.output.HAS_TO_SAVE = False

    def test_init_jet(self):
        pass


class TestSolverNS2DInitDipole(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.init_fields.type = "dipole"
        params.output.HAS_TO_SAVE = False

    def test_init_dipole(self):
        pass


if __name__ == "__main__":
    unittest.main()
