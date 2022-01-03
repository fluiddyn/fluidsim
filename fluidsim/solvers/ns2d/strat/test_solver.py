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
    def init_params(cls):
        params = super().init_params()

        params.forcing.enable = True
        params.forcing.type = "tcrandom_anisotropic"
        params.forcing.nkmin_forcing = 4
        params.forcing.nkmax_forcing = 6
        params.forcing.key_forced = "ap_fft"
        params.forcing.tcrandom_anisotropic.kz_negative_enable = True
        return params

    def test_forcing_linear_mode(self):
        sim = self.sim
        sim.time_stepping.start()
        if mpi.nb_proc == 1:
            sim.forcing.forcing_maker.plot_forcing_region()


class TestForcingLinearModeBis(TestForcingLinearMode):
    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.forcing.tcrandom_anisotropic.delta_angle = np.pi / 8


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
            periods[key] = 0.1

        params.output.ONLINE_PLOT_OK = True
        params.output.periods_print.print_stdout = 0.2
        params.output.periods_plot.phys_fields = 0.2

        Lx, Ly = params.oper.Lx, params.oper.Ly
        nx, ny = params.oper.nx, params.oper.ny
        params.output.temporal_spectra.probes_region = (0.0, Lx, 0.0, Ly)
        params.output.temporal_spectra.probes_deltax = Lx / nx
        params.output.temporal_spectra.probes_deltay = Ly / ny
        params.output.temporal_spectra.SAVE_AS_FLOAT32 = True

        params.output.spatiotemporal_spectra.probes_region = (
            nx // 2 + 1,
            ny // 2 + 1,
        )
        params.output.spatiotemporal_spectra.SAVE_AS_COMPLEX64 = False

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

        if mpi.nb_proc > 1:
            plt.close("all")
            return

        sim3.output.phys_fields.animate(
            "ux",
            dt_frame_in_sec=1e-6,
            dt_equations=0.3,
            repeat=False,
            clim=(-1, 1),
            save_file=False,
            numfig=1,
        )

        sim2 = fls.load_sim_for_plot(sim.output.path_run)
        sim2.output

        sim2.output.spect_energy_budg.load_mean()

        means = sim2.output.spatial_means.load()
        # `:-1` because the last time is saved twice in spatial_means
        # (see SpatialMeansBase.__init__)
        energy_K_mean = means["EK"][:-1].mean()

        sim2.output.temporal_spectra.load_time_series()
        spectra_omega = sim2.output.temporal_spectra.compute_spectra()
        omegas = spectra_omega["omegas"]
        delta_omega = omegas[1]

        sim2.output.temporal_spectra.plot_spectra()

        spatiotemporal_spectra = sim2.output.spatiotemporal_spectra
        series_kxky = spatiotemporal_spectra.load_time_series()

        spectra_kxkyomega = spatiotemporal_spectra.compute_spectra()
        spectra_omega_from_spatiotemp = (
            spatiotemporal_spectra.compute_temporal_spectra()
        )

        deltakx = 2 * np.pi / self.params.oper.Lx
        order = spectra_kxkyomega["dims_order"]
        KX = deltakx * spectra_kxkyomega[f"K{order[1]}_adim"]
        kx_max = self.params.oper.nx // 2 * deltakx

        assert kx_max == KX.max()

        from fluidsim.solvers.ns2d.output.spatiotemporal_spectra import (
            _sum_wavenumber2D,
        )

        def sum_wavenumber(field):
            return _sum_wavenumber2D(field, KX, kx_max)

        tspectrum_mean = (
            spectra_omega["spectrum_ux"] + spectra_omega["spectrum_uy"]
        )

        energy_tspect_mean = 0.5 * delta_omega * tspectrum_mean.sum()

        assert np.allclose(energy_K_mean, energy_tspect_mean), (
            energy_K_mean / energy_tspect_mean
        )

        _ = spatiotemporal_spectra.save_spectra_kzkhomega()
        spectra_kzkhomega = spatiotemporal_spectra.load_spectra_kzkhomega()

        delta_kz = spectra_kzkhomega["kz_spectra"][1]
        delta_kh = spectra_kzkhomega["kh_spectra"][1]
        delta_omega = spectra_kzkhomega["omegas"][1]
        coef = delta_kz * delta_kh * delta_omega

        for letter in "xy":
            vi_fft = series_kxky[f"u{letter}_Fourier"]
            spectrum_kxkykzomega = spectra_kxkyomega["spectrum_u" + letter]
            spectrum_omega = spectra_omega["spectrum_u" + letter]
            spectrum_omega_from_spatiotemp = spectra_omega_from_spatiotemp[
                "spectrum_u" + letter
            ]
            spectrum_kzkhomega = spectra_kzkhomega["spectrum_u" + letter]

            E_series_kxkykz = 0.5 * sum_wavenumber(
                (abs(vi_fft) ** 2).mean(axis=-1)
            )
            assert E_series_kxkykz > 0, (letter, vi_fft)

            E_kxkykzomega = (
                0.5
                * delta_omega
                * sum_wavenumber(spectrum_kxkykzomega.sum(axis=-1))
            )
            E_omega = 0.5 * delta_omega * spectrum_omega.sum()
            E_omega_from_spatiotemp = (
                0.5 * delta_omega * spectrum_omega_from_spatiotemp.sum()
            )

            E_kzkhomega = 0.5 * coef * spectrum_kzkhomega.sum()

            assert np.allclose(E_omega, E_kxkykzomega), (
                letter,
                E_kxkykzomega / E_omega,
            )

            assert np.allclose(E_series_kxkykz, E_kxkykzomega), (
                letter,
                E_kxkykzomega / E_series_kxkykz,
            )

            assert np.allclose(E_series_kxkykz, E_kzkhomega), (
                letter,
                E_kzkhomega / E_series_kxkykz,
            )

            assert np.allclose(E_omega, E_omega_from_spatiotemp), (
                letter,
                E_omega,
                E_omega_from_spatiotemp,
            )

        sim2.output.spatiotemporal_spectra.plot_kzkhomega(key_field="ux")
        sim2.output.spatiotemporal_spectra.plot_kzkhomega(
            key_field="ux", equation="omega=N/2"
        )
        sim2.output.spatiotemporal_spectra.plot_kzkhomega(
            key_field="ux", equation="kh=1"
        )
        sim2.output.spatiotemporal_spectra.plot_kzkhomega(
            key_field="ux", equation="kz=1"
        )
        sim2.output.spatiotemporal_spectra.plot_kzkhomega(
            key_field="ux", equation="ikh=1"
        )
        sim2.output.spatiotemporal_spectra.plot_kzkhomega(
            key_field="ux", equation="ikz=1"
        )
        sim2.output.spatiotemporal_spectra.plot_temporal_spectra()

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
