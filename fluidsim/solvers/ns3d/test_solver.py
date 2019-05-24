import unittest
import sys

import numpy as np
import matplotlib.pyplot as plt

import fluidsim as fls

import fluiddyn.util.mpi as mpi

from fluidsim.solvers.ns3d.solver import Simul

from fluidsim.base.output import run

from fluidsim.util.testing import TestSimul


class TestSimulBase(TestSimul):
    Simul = Simul

    @classmethod
    def init_params(cls):

        params = cls.params = cls.Simul.create_default_params()

        params.short_name_type_run = "test"
        params.output.sub_directory = "unittests"
        nx = 32
        params.oper.nx = nx
        params.oper.ny = nx
        params.oper.nz = nx

        Lx = 6.0
        params.oper.Lx = Lx
        params.oper.Ly = Lx
        params.oper.Lz = Lx

        params.oper.coef_dealiasing = 2.0 / 3
        params.nu_8 = 2.0

        params.time_stepping.t_end = 0.2

        return params


class TestTendency(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()

        params.init_fields.type = "noise"
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
        params.init_fields.type = "dipole"

        params.time_stepping.max_elapsed = 600

        params.forcing.enable = True
        params.forcing.type = "in_script"
        params.forcing.key_forced = "vx_fft"

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.1

    def test_output(self):

        sim = self.sim
        sim.time_stepping.start()

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

        sim.output.phys_fields._get_grid1d("iz=0")
        sim.output.phys_fields._get_grid1d("iy=0")

        path_run = sim.output.path_run
        if mpi.nb_proc > 1:
            path_run = mpi.comm.bcast(path_run)

        if mpi.nb_proc == 1:
            sim2 = fls.load_sim_for_plot(path_run)
            sim2.output.print_stdout.load()
            sim2.output.print_stdout.plot()
            sim2.output.spatial_means.load()
            sim2.output.spatial_means.load_dataset()
            sim2.output.spatial_means.plot()
            sim2.output.spectra.load1d_mean()
            sim2.output.spectra.load3d_mean()
            sim2.output.spectra.plot1d(
                tmin=0.1,
                tmax=10,
                delta_t=0.01,
                coef_compensate=5 / 3,
                coef_plot_k3=1.0,
                coef_plot_k53=1.0,
            )
            sim2.output.spectra.plot3d(
                tmin=0.1,
                tmax=10,
                delta_t=0.01,
                coef_compensate=5 / 3,
                coef_plot_k3=1.0,
                coef_plot_k53=1.0,
            )

            sim2.output.phys_fields.set_equation_crosssection(
                f"x={sim.oper.Lx/4}"
            )
            sim2.output.phys_fields.animate("vx")

            sim2.output.phys_fields.plot(
                field="vx", time=10, equation=f"z={sim.oper.Lz/4}"
            )

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


if __name__ == "__main__":
    unittest.main()
