import unittest

import matplotlib.pyplot as plt

import fluiddyn.util.mpi as mpi

import fluidsim as fls

from ..test_solver import TestSimulBase as _Base, classproperty


class TestSimulBase(_Base):
    @classproperty
    def Simul(self):
        from .solver import Simul

        return Simul


class TestOutput(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()
        params.init_fields.type = "dipole"

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.2

    def test_output(self):

        sim = self.sim
        sim.time_stepping.start()

        if mpi.nb_proc > 1:
            return

        sim2 = fls.load_sim_for_plot(sim.output.path_run)
        sim2.output.print_stdout.load()
        sim2.output.print_stdout.plot()
        sim2.output.spatial_means.load()
        sim2.output.spatial_means.load_dataset()
        sim2.output.spatial_means.plot()
        sim2.output.spectra.load1d_mean()
        sim2.output.spectra.load3d_mean()
        sim2.output.spectra.plot1d_times(
            tmin=0.1,
            tmax=10,
            delta_t=0.01,
            coef_compensate=5 / 3,
            coef_plot_k3=1.0,
            coef_plot_k53=1.0,
        )
        sim2.output.spectra.plot3d_times(
            tmin=0.1,
            tmax=10,
            delta_t=0.01,
            coef_compensate=5 / 3,
            coef_plot_k3=1.0,
            coef_plot_k53=1.0,
        )

        sim2.output.phys_fields.set_equation_crosssection(f"x={sim.oper.Lx/4}")
        sim2.output.phys_fields.animate("vx")

        sim2.output.phys_fields.plot(field="vx", time=10)

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
