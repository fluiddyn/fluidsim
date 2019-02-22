import unittest

import numpy as np
import matplotlib.pyplot as plt

import fluidsim as fls

import fluiddyn.util.mpi as mpi

from fluidsim.solvers.ns2d.strat.solver import Simul
from fluidsim.solvers.ns2d.test_solver import TestSimulBase as Base


class TestSimulBase(Base):
    Simul = Simul


class TestSolverNS2DTendency(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.time_stepping.USE_CFL = False
        params.time_stepping.USE_T_END = False
        # todo: understand why it fails for larger it_end and deltat0. I (pa)
        # guess there could be a bug hidden.
        params.time_stepping.it_end = 2
        params.time_stepping.deltat0 = 0.02
        params.output.HAS_TO_SAVE = False

    def test_tendency(self):
        sim = self.sim

        rot_fft = sim.state.get_var("rot_fft")
        b_fft = sim.state.get_var("b_fft")

        assert not np.any(b_fft)

        tend = sim.tendencies_nonlin(state_spect=sim.state.state_spect)
        Frot_fft = tend.get_var("rot_fft")
        Fb_fft = tend.get_var("b_fft")

        T_rot = np.real(Frot_fft.conj() * rot_fft)

        ratio = sim.oper.sum_wavenumbers(T_rot) / sim.oper.sum_wavenumbers(
            abs(T_rot)
        )

        self.assertGreater(1e-15, ratio)

        # init_fields = noise gives energy only to rot
        sim.time_stepping.start()

        rot_fft = sim.state.get_var("rot_fft")
        b_fft = sim.state.get_var("b_fft")

        # init_fields = noise gives energy only to rot
        assert np.any(b_fft)

        tend = sim.tendencies_nonlin(state_spect=sim.state.state_spect)
        Frot_fft = tend.get_var("rot_fft")
        Fb_fft = tend.get_var("b_fft")

        assert sim.check_energy_conservation(rot_fft, b_fft, Frot_fft, Fb_fft)


class TestForcingLinearMode(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()

        params.forcing.enable = True
        params.forcing.type = "tcrandom_anisotropic"
        params.forcing.nkmin_forcing = 4
        params.forcing.nkmax_forcing = 6
        params.forcing.key_forced = "ap_fft"

    def test_forcing_linear_mode(self):
        sim = self.sim
        sim.time_stepping.start()
        if mpi.nb_proc == 1:
            sim.forcing.forcing_maker.plot_forcing_region()


class TestForcingConstantRateEnergy(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.forcing.enable = True
        params.forcing.type = "tcrandom"
        params.forcing.normalized.constant_rate_of = "energy"
        params.forcing.key_forced = "rot_fft"
        params.forcing.tcrandom_anisotropic.kz_negative_enable = True
        params.forcing.forcing_rate = 3.333

        params.output.periods_save.spatial_means = 1e-6

        return params

    def test_(self):
        self.sim.time_stepping.start()

        if mpi.rank == 0:
            # Does the energy injection rate have the correct value at all times ?
            means = self.sim.output.spatial_means.load()
            P_tot = means["PK_tot"] + means["PA_tot"]
            assert np.allclose(P_tot, self.sim.params.forcing.forcing_rate)


class TestForcingConstantRateEnergyAP(TestForcingConstantRateEnergy):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.forcing.key_forced = "ap_fft"
        params.forcing.tcrandom_anisotropic.kz_negative_enable = False


class TestForcingOutput(TestSimulBase):
    @classmethod
    def init_params(self):

        params = super().init_params()

        # Time stepping parameters
        params.time_stepping.USE_CFL = False
        params.time_stepping.USE_T_END = False
        params.time_stepping.it_end = 2
        params.time_stepping.deltat0 = 0.1

        params.forcing.enable = True
        # params.forcing.type = "tcrandom"
        # Forcing also linear mode!!!
        params.forcing.type = "tcrandom_anisotropic"
        params.forcing.nkmin_forcing = 4
        params.forcing.nkmax_forcing = 6

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.2

        params.output.ONLINE_PLOT_OK = True
        params.output.periods_print.print_stdout = 0.2
        params.output.periods_plot.phys_fields = 0.2

        # Spatio-temporal spectra
        params.output.spatio_temporal_spectra.size_max_file = 0.01
        params.output.spatio_temporal_spectra.time_decimate = 1
        params.output.spatio_temporal_spectra.spatial_decimate = 1
        params.output.spatio_temporal_spectra.time_start = 0

        params.output.frequency_spectra.size_max_file = 0.004
        params.output.frequency_spectra.time_start = 0
        params.output.frequency_spectra.time_decimate = 1
        params.output.frequency_spectra.spatial_decimate = 2

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

        sim.output.compute_enstrophy()

        if mpi.nb_proc == 1:

            plt.close("all")

            sim.output.frequency_spectra.compute_frequency_spectra()

            sim.output.plot_summary()

            sim.output.spectra.plot2d()

            sim.output.spatial_means.plot_energy()
            sim.output.spatial_means.plot_dt_energy()
            sim.output.spatial_means.plot_energy_shear_modes()

            plt.close("all")
            sim.output.spatial_means.compute_time_means()
            sim.output.spatial_means.load_dataset()
            sim.output.spatial_means.time_first_saved()
            sim.output.spatial_means.time_last_saved()

            sim.output.spectra_multidim.plot()

            with self.assertRaises(ValueError):
                sim.state.get_var("test")

            sim2 = fls.load_sim_for_plot(sim.output.path_run)
            sim2.output

            spatio_temporal_spectra = sim2.output.spatio_temporal_spectra
            spatio_temporal_spectra.compute_frequency_spectra()
            spatio_temporal_spectra.print_info_frequency_spectra()
            spatio_temporal_spectra.plot_frequency_spectra_individual_mode(
                mode=(1, 1)
            )
            spatio_temporal_spectra.plot_kx_omega_cross_section()
            spatio_temporal_spectra.plot_kz_omega_cross_section()

            sim2.output.increments.load()
            sim2.output.increments.plot()
            sim2.output.increments.load_pdf_from_file()

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


class TestSolverNS2DInitLinearMode(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.init_fields.type = "linear_mode"
        params.output.HAS_TO_SAVE = False

    def test_init_linear_mode(self):
        pass


if __name__ == "__main__":
    unittest.main()
