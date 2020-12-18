import unittest
import sys
from pathlib import Path

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

    @classmethod
    def init_params(cls):

        params = cls.params = cls.Simul.create_default_params()

        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"
        nx = 32
        params.oper.nx = nx
        params.oper.ny = nx * 3 // 4

        Lx = 6.0
        params.oper.Lx = Lx
        params.oper.Ly = Lx * 3 // 4
        try:
            params.oper.nz = nx // 2
            params.oper.Lz = Lx // 2
        except AttributeError:
            pass

        params.oper.coef_dealiasing = 2.0 / 3
        params.nu_4 = 2.0
        params.nu_8 = 2.0

        params.time_stepping.t_end = 0.2
        params.init_fields.type = "noise"

        return params


class TestTendency(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
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
    def init_params(self):
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

    def test_output(self):

        sim = self.sim
        sim.time_stepping.start()

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

        # compute twice for better coverage
        sim.state.compute("rotz")
        sim.state.compute("rotz")
        sim.state.compute("divh")

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

        if mpi.nb_proc == 1:
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
        params, Simul = load_for_restart(sim.output.path_run)
        params.time_stepping.t_end += 2.0
        sim_restart = Simul(params)
        sim_restart.time_stepping.start()

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
            assert np.allclose(np.mean(var ** 2), np.mean(var_big ** 2))


class TestForcingMilestone(TestSimulBase):
    @classmethod
    def init_params(self):
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
    def init_params(self):
        params = super().init_params()
        params.time_stepping.t_end = 2.0
        params.forcing.milestone.nx_max = 24
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
