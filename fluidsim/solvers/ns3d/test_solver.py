import unittest
import sys
from pathlib import Path
from math import pi

import pytest

import numpy as np
import matplotlib.pyplot as plt

import fluidsim as fls

import fluiddyn.util.mpi as mpi


from fluidsim import (
    modif_resolution_from_dir,
    load_state_phys_file,
    load_for_restart,
)
from fluidsim.base.output import run


from fluidsim.util.testing import TestSimul, skip_if_no_fluidfft, classproperty


@skip_if_no_fluidfft
class TestSimulBase(TestSimul):
    @classproperty
    def Simul(cls):
        from fluidsim.solvers.ns3d.solver import Simul

        return Simul

    @staticmethod
    def _init_grid(params, nx):
        params.oper.nx = nx
        params.oper.ny = nx * 3 // 4
        try:
            params.oper.nz = nx // 2
        except AttributeError:
            pass

    @classmethod
    def init_params(cls):

        params = cls.params = cls.Simul.create_default_params()

        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"
        cls._init_grid(params, nx=16)

        Lx = 6.0
        params.oper.Lx = Lx
        params.oper.Ly = Lx * params.oper.ny / params.oper.nx
        try:
            params.oper.Lz = Lx * params.oper.nz / params.oper.nx
        except AttributeError:
            pass

        params.oper.coef_dealiasing = 2.0 / 3
        params.nu_4 = 2.0
        params.nu_8 = 2.0

        params.time_stepping.t_end = 1.5 * params.time_stepping.deltat_max
        params.init_fields.type = "noise"

        return params


class TestTendency(TestSimulBase):
    @classmethod
    def init_params(cls):
        params = super().init_params()
        cls._init_grid(params, nx=20)
        params.output.HAS_TO_SAVE = False

    def test_tendency(self):

        sim = self.sim
        tend = sim.tendencies_nonlin(state_spect=sim.state.state_spect)

        T_tot = np.ascontiguousarray(sim.oper.create_arrayK(value=0.0).real)

        for axis in ("x", "y", "z"):
            key = f"v{axis}_fft"
            vi_fft = sim.state.get_var(key)
            Fi_fft = tend.get_var(key)
            T_tot += np.real(Fi_fft.conj() * vi_fft)

        ratio = sim.oper.sum_wavenumbers(T_tot) / sim.oper.sum_wavenumbers(
            abs(T_tot)
        )

        self.assertGreater(1e-15, abs(ratio))


class TestOutput(TestSimulBase):
    @classmethod
    def init_params(cls):
        params = super().init_params()

        params.oper.truncation_shape = "no_multiple_aliases"

        params.init_fields.type = "dipole"

        params.time_stepping.max_elapsed = 600
        params.time_stepping.type_time_scheme = "RK2_phaseshift_random"

        params.forcing.enable = True
        params.forcing.type = "in_script"
        params.forcing.key_forced = "vx_fft"

        params.nu_m4 = 1e-3

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.1

        Lx, Ly, Lz = params.oper.Lx, params.oper.Ly, params.oper.Lz
        nx, ny, nz = params.oper.nx, params.oper.ny, params.oper.nz
        probes_region = (0.0, Lx, 0.0, Ly, 0.0, Lz)
        params.output.temporal_spectra.probes_region = probes_region
        params.output.temporal_spectra.probes_deltax = Lx / nx
        params.output.temporal_spectra.probes_deltay = Ly / ny
        params.output.temporal_spectra.probes_deltaz = Lz / nz
        params.output.temporal_spectra.SAVE_AS_FLOAT32 = False

        nx, ny, nz = params.oper.nx, params.oper.ny, params.oper.nz
        params.output.spatiotemporal_spectra.probes_region = (
            nx // 2,
            ny // 2,
            nz // 2,
        )
        params.output.spatiotemporal_spectra.SAVE_AS_COMPLEX64 = False

    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
    def test_output(self):

        sim = self.sim

        # put energy in vz
        vz = sim.state.state_phys.get_var("vz")
        X, Y, Z = sim.oper.get_XYZ_loc()
        vz += 0.05 * np.cos(2 * pi * X / sim.oper.Lx)
        sim.state.statespect_from_statephys()

        sim.state.check_energy_equal_phys_spect()
        sim.time_stepping.start()
        sim.state.check_energy_equal_phys_spect()

        # testing phaseshift
        phaseshift = sim.time_stepping._get_phaseshift()
        assert phaseshift.shape == sim.oper.Kx.shape
        assert sim.time_stepping._get_phaseshift() is phaseshift

        if mpi.nb_proc == 1:

            phys_fields = sim.output.phys_fields
            phys_fields.plot(equation=f"iz=0", numfig=1000)

            phys_fields.get_field_to_plot_from_state()
            phys_fields.get_field_to_plot_from_state(sim.state.get_var("vx"))
            phys_fields.get_field_to_plot_from_state("vx", "x=0")
            phys_fields.get_field_to_plot_from_state("vx", "ix=0")
            phys_fields.get_field_to_plot_from_state("vx", "y=0")
            phys_fields.get_field_to_plot_from_state("vx", "iy=0")
            phys_fields.get_field_to_plot_from_state("vx", "z=0")

        with pytest.raises(ValueError):
            sim.state.compute("foo")
        sim.state.compute("foo", RAISE_ERROR=False)
        for key in ("divh", "vt", "vp", "rotz"):
            sim.state.compute(key)
        # compute twice for better coverage
        sim.state.compute("rotz")

        sim.output.phys_fields._get_grid1d("iz=0")
        sim.output.phys_fields._get_grid1d("iy=0")

        path_run = sim.output.path_run
        if mpi.nb_proc > 1:
            path_run = mpi.comm.bcast(path_run)

        if mpi.nb_proc == 1:
            sim2 = fls.load_sim_for_plot(path_run)
            sim2.output.plot_summary(tmin=0, key_field="vx")
            sim2.output.print_stdout.load()
            sim2.output.print_stdout.plot()
            sim2.output.spatial_means.load()
            sim2.output.spatial_means.load_dataset()
            sim2.output.spatial_means.plot(plot_injection=True, plot_hyper=True)
            sim2.output.spatial_means.plot_dt_E()
            sim2.output.spectra.load1d_mean()
            sim2.output.spectra.load3d_mean()
            sim2.output.spectra.plot1d(
                coef_plot_k2=1.0,
                coef_plot_k3=1.0,
                coef_plot_k53=1.0,
                xlim=(0.1, 1),
                ylim=(0.1, 1),
            )
            sim2.output.spectra.plot1d_times(
                tmin=0.1,
                tmax=10,
                delta_t=0.01,
                coef_compensate=5 / 3,
                coef_plot_k2=1.0,
                coef_plot_k3=1.0,
                coef_plot_k53=1.0,
                xlim=(0.1, 1),
                ylim=(0.1, 1),
            )
            sim2.output.spectra.plot3d_times(
                tmin=0.1,
                tmax=10,
                coef_compensate=5 / 3,
                coef_plot_k2=1.0,
                coef_plot_k3=1.0,
                coef_plot_k53=1.0,
            )
            sim2.output.spectra.plot3d_cumul_diss(tmin=0.1, tmax=10)

            sim2.output.phys_fields.set_equation_crosssection(
                f"x={sim.oper.Lx/4}"
            )
            sim2.output.phys_fields.animate("vx")

            sim2.output.phys_fields.plot(
                field="vx", time=10, equation=f"z={sim.oper.Lz/4}"
            )
            sim2.plot_freq_diss("z")

        sim3 = fls.load_state_phys_file(path_run, modif_save_params=False)
        sim3.params.time_stepping.t_end += 0.2
        sim3.time_stepping.start()

        if mpi.nb_proc > 1:
            plt.close("all")
            return

        sim3.output.phys_fields.animate(
            "vx",
            dt_frame_in_sec=1e-6,
            dt_equations=0.3,
            repeat=False,
            clim=(-1, 1),
            save_file=False,
            numfig=1,
        )

        sys.argv = ["fluidsim-create-xml-description", path_run]
        run()
        sim3.output.temporal_spectra.plot_spectra()
        sim3.output.temporal_spectra.save_data_as_phys_fields(delta_index_times=2)
        sim3.output.temporal_spectra.save_spectra()

        t_end = sim3.params.time_stepping.t_end

        spatiotemporal_spectra = sim3.output.spatiotemporal_spectra
        series_kxkykz = spatiotemporal_spectra.load_time_series(tmax=t_end)

        spectra_kxkykzomega = spatiotemporal_spectra.compute_spectra(tmax=t_end)
        spectra_omega_from_spatiotemp = (
            spatiotemporal_spectra.compute_temporal_spectra(tmax=t_end)
        )
        spectra_omega = sim3.output.temporal_spectra.compute_spectra()

        means = sim3.output.spatial_means.load()

        deltakx = 2 * pi / self.params.oper.Lx
        order = spectra_kxkykzomega["dims_order"]
        KX = deltakx * spectra_kxkykzomega[f"K{order[2]}_adim"]
        kx_max = self.params.oper.nx // 2 * deltakx

        assert kx_max == KX.max()

        from fluidsim.solvers.ns3d.output.spatiotemporal_spectra import (
            _sum_wavenumber3D,
        )

        def sum_wavenumber(field):
            return _sum_wavenumber3D(field, KX, kx_max)

        _ = spatiotemporal_spectra.save_spectra_kzkhomega(
            save_urud=True, tmax=t_end
        )
        spectra_kzkhomega = spatiotemporal_spectra.load_spectra_kzkhomega(
            save_urud=True, tmax=t_end
        )

        delta_kz = spectra_kzkhomega["kz_spectra"][1]
        delta_kh = spectra_kzkhomega["kh_spectra"][1]
        delta_omega = spectra_kzkhomega["omegas"][1]
        coef = delta_kz * delta_kh * delta_omega

        for letter in "xyz":
            vi_fft = series_kxkykz[f"v{letter}_Fourier"]
            spectrum_kxkykzomega = spectra_kxkykzomega["spectrum_v" + letter]
            spectrum_omega = spectra_omega["spectrum_v" + letter]
            spectrum_omega_from_spatiotemp = spectra_omega_from_spatiotemp[
                "spectrum_v" + letter
            ]
            spectrum_kzkhomega = spectra_kzkhomega["spectrum_v" + letter]

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

            # `:-1` because the last time is saved twice in spatial_means
            # (see SpatialMeansBase.__init__)
            E_mean = means["E" + letter][:-1].mean()

            assert np.allclose(E_omega, E_kxkykzomega), (
                letter,
                E_kxkykzomega / E_mean,
            )

            assert np.allclose(E_series_kxkykz, E_kxkykzomega), (
                letter,
                E_kxkykzomega / E_series_kxkykz,
            )

            assert np.allclose(E_series_kxkykz, E_kzkhomega), (
                letter,
                E_kzkhomega / E_series_kxkykz,
            )

            assert np.allclose(E_mean, E_series_kxkykz), (
                letter,
                E_series_kxkykz / E_mean,
            )

            assert np.allclose(E_omega, E_omega_from_spatiotemp), (
                letter,
                E_omega,
                E_omega_from_spatiotemp,
            )

            # print(f"{spectrum_omega.sum() / spectrum_omega_from_spatiotemp.sum() = }")

            # this condition is not exactly fulfilled (why?)
            # assert np.allclose(
            #     spectrum_omega, spectrum_omega_from_spatiotemp
            # ), (
            #     letter,
            #     spectrum_omega,
            #     spectrum_omega_from_spatiotemp,
            #     spectrum_omega / spectrum_omega_from_spatiotemp,
            #     spectrum_omega.sum() / spectrum_omega_from_spatiotemp.sum(),
            # )

        spectrum_Khd = spectra_kzkhomega["spectrum_Khd"]
        spectrum_vz = spectra_kzkhomega["spectrum_vz"]

        # because k \cdot \hat v = 0, for kz = 0, Khd = 0
        assert np.allclose(spectrum_Khd[0].sum(), 0.0)

        # because k \cdot \hat v = 0, for kh = 0, Kz = 0
        # `1:` because the energy in the mode kx=ky=kz=0 is not zero.
        assert np.allclose(spectrum_vz[1:, 0, :].sum(), 0.0)

        sim3.output.spatiotemporal_spectra.plot_temporal_spectra()
        sim3.output.spatiotemporal_spectra.plot_kzkhomega(
            key_field="Khr", equation="kh=1"
        )

        plt.close("all")


class TestInitInScript(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.output.HAS_TO_SAVE = False
        params.init_fields.type = "in_script"

    def test_init_in_script(self):

        sim = self.sim

        # here we have to initialize the flow fields

        variables = {
            k: 1e-2 * sim.oper.create_arrayX_random() for k in ("vx", "vy", "vz")
        }

        X, Y, Z = sim.oper.get_XYZ_loc()

        sim.state.init_statephys_from(**variables)

        sim.state.statespect_from_statephys()
        sim.state.statephys_from_statespect()

        vx_fft = sim.state.get_var("vx_fft")
        vy_fft = sim.state.get_var("vy_fft")
        vz_fft = sim.state.get_var("vz_fft")

        sim.state.init_from_vxvyfft(vx_fft, vy_fft)
        sim.state.init_from_vxvyvzfft(vx_fft, vy_fft, vz_fft)
        sim.state.check_energy_equal_phys_spect()


class TestForcingTaylorGreen(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.nu_2 = 0.001
        params.init_fields.type = "noise"
        params.init_fields.noise.velo_max = 0.001
        params.forcing.enable = True
        params.forcing.type = "taylor_green"
        params.forcing.taylor_green.amplitude = 1.0
        params.output.periods_save.phys_fields = 2.0

    def test_forcing(self):
        sim = self.sim
        sim.time_stepping.start()
        sim.state.check_energy_equal_phys_spect()


class TestForcingTimeCorrelatedRandomPseudoSpectralAnisotropic3D(TestSimulBase):
    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.nu_2 = 0.001
        params.projection = "poloidal"
        params.init_fields.type = "noise"
        params.init_fields.noise.velo_max = 0.001
        params.forcing.enable = True
        params.forcing.type = "tcrandom_anisotropic"
        params.forcing.forcing_rate = 1.0
        params.forcing.key_forced = "vp_fft"

        # keep these values (catching a potential bug #93)
        params.forcing.nkmin_forcing = 0.9
        params.forcing.nkmax_forcing = 1.5
        params.forcing.tcrandom_anisotropic.angle = np.pi / 4
        params.forcing.tcrandom_anisotropic.delta_angle = np.pi / 6
        params.forcing.tcrandom_anisotropic.kz_negative_enable = True
        params.forcing.tcrandom.time_correlation = 1.0
        return params

    def test_forcing(self):
        sim = self.sim
        params = sim.params

        sim.time_stepping.main_loop(print_begin=True, save_init_field=True)

        for key_forced in ("vt_fft", "rotz_fft", "divh_fft"):
            params.time_stepping.t_end += 0.2
            params.forcing.key_forced = (
                sim.forcing.forcing_maker.key_forced
            ) = key_forced
            sim.time_stepping.init_from_params()
            sim.time_stepping.main_loop()

        sim.time_stepping.finalize_main_loop()
        sim.state.check_energy_equal_phys_spect()

        if mpi.nb_proc == 1:
            sim.forcing.forcing_maker.plot_forcing_region()

        with pytest.raises(ValueError):
            sim.params.projection = "popo"
            sim._init_projection()


class TestForcingTimeCorrelatedRandomPseudoSpectralAnisotropic3DBis(
    TestForcingTimeCorrelatedRandomPseudoSpectralAnisotropic3D
):
    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.forcing.tcrandom_anisotropic.delta_angle = None
        params.projection = "poloidal"
        params.forcing.key_forced = "vt_fft"

    def test_forcing(self):
        sim = self.sim
        if mpi.nb_proc == 1:
            sim.forcing.forcing_maker.plot_forcing_region()
        sim.time_stepping.start()
        sim.state.check_energy_equal_phys_spect()


class TestForcingWatuCoriolis(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()

        params.init_fields.type = "noise"

        omega_f = 0.3  # rad/s
        delta_omega_f = 0.03  # rad/s
        N = 0.4  # rad/s
        amplitude = 0.05  # m
        period_N = 2 * np.pi / N
        period_forcing = 1e1 * period_N

        params.forcing.enable = True
        params.forcing.type = "watu_coriolis"

        watu = params.forcing.watu_coriolis
        watu.omega_f = omega_f
        watu.delta_omega_f = delta_omega_f
        watu.amplitude = amplitude
        watu.period_forcing = period_forcing
        watu.approximate_dt = period_N / 1e1
        watu.nb_wave_makers = 2

        params.output.periods_save.phys_fields = 2.0

    def test_forcing(self):

        sim = self.sim
        sim.time_stepping.start()
        sim.state.check_energy_equal_phys_spect()

        params, Simul = load_for_restart(sim.output.path_run)
        params.time_stepping.t_end += 2.0
        sim_restart = Simul(params)
        sim_restart.time_stepping.start()
        sim_restart.state.check_energy_equal_phys_spect()

        if mpi.nb_proc > 1:
            return

        modif_resolution_from_dir(
            self.sim.output.path_run, coef_modif_resol=3.0 / 2, PLOT=True
        )

        path_dir_big = next(Path(self.sim.output.path_run).glob("State_phys_*"))
        sim_big = load_state_phys_file(path_dir_big)

        for key in self.sim.state.keys_state_phys:
            var = sim_restart.state.get_var(key)
            var_big = sim_big.state.get_var(key)
            assert np.allclose(np.mean(var**2), np.mean(var_big**2))


class TestForcingMilestone(TestSimulBase):
    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.forcing.enable = True
        params.forcing.type = "milestone"
        movement = params.forcing.milestone.movement
        movement.type = "uniform"
        movement.uniform.speed = 1.0

        params.init_fields.type = "noise"
        params.init_fields.noise.velo_max = 5e-3

        return params

    def test_milestone(self):
        self.sim.time_stepping.start()


class TestForcingMilestonePeriodicUniform(TestForcingMilestone):
    @classmethod
    def init_params(cls):
        params = super().init_params()
        params.oper.NO_SHEAR_MODES = True
        params.time_stepping.t_end = 2.0
        params.forcing.milestone.nx_max = 16
        movement = params.forcing.milestone.movement
        movement.type = "periodic_uniform"
        movement.periodic_uniform.length = 2.0
        movement.periodic_uniform.length_acc = 0.25
        movement.periodic_uniform.speed = 2.5

    def test_milestone(self):
        super().test_milestone()
        self.sim.forcing.get_info()


if __name__ == "__main__":
    unittest.main()
