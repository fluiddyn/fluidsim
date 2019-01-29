import unittest

import numpy as np
import matplotlib.pyplot as plt

import fluiddyn.util.mpi as mpi

import fluidsim as fls

from ..test_solver import TestSimulBase as _Base

from .solver import Simul


class TestSimulBase(_Base):
    Simul = Simul


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

        key = "b_fft"
        b_fft = sim.state.get_var(key)
        F_fft = tend.get_var(key)

        T_tot += 1.0 / sim.params.N ** 2 * np.real(F_fft.conj() * b_fft)

        ratio = sim.oper.sum_wavenumbers(T_tot) / sim.oper.sum_wavenumbers(
            abs(T_tot)
        )

        self.assertGreater(1e-15, abs(ratio))


class TestOutput(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()

        params.init_fields.type = "dipole"

        params.forcing.enable = True
        params.forcing.type = "in_script"

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.2

        for tag in params.output._tag_children:
            if tag.startswith("periods"):
                continue
            child = params.output[tag]
            if hasattr(child, "HAS_TO_PLOT_SAVED"):
                child["HAS_TO_PLOT_SAVED"] = True

    def test_output(self):

        sim = self.sim

        oper = sim.oper
        X, Y, Z = oper.get_XYZ_loc()

        def compute_forcing_fft_each_time(self):
            return {"vx_fft": oper.create_arrayK(value=0)}

        sim.forcing.forcing_maker.monkeypatch_compute_forcing_fft_each_time(
            compute_forcing_fft_each_time
        )

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
        # here we have to initialize the flow fields
        sim = self.sim
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
