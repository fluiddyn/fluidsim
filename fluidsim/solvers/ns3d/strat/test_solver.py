import unittest

import pytest

import numpy as np
import matplotlib.pyplot as plt

import fluiddyn.util.mpi as mpi

import fluidsim as fls

from fluidsim.extend_simul import extend_simul_class

from ..test_solver import TestSimulBase as _Base, classproperty


class TestSimulBase(_Base):
    @classproperty
    def Simul(cls):
        from .solver import Simul

        return Simul


class TestTendency(TestSimulBase):
    @classmethod
    def init_params(cls):
        params = super().init_params()
        cls._init_grid(params, nx=64)

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

        T_tot += 1.0 / sim.params.N**2 * np.real(F_fft.conj() * b_fft)

        ratio = sim.oper.sum_wavenumbers(T_tot) / sim.oper.sum_wavenumbers(
            abs(T_tot)
        )

        self.assertGreater(1e-15, abs(ratio))


class TestOutput(TestSimulBase):
    @classproperty
    def Simul(cls):
        from .solver import Simul
        from fluidsim.extend_simul.spatial_means_regions_milestone import (
            SpatialMeansRegions,
        )

        return extend_simul_class(Simul, SpatialMeansRegions)

    @classmethod
    def init_params(self):
        params = super().init_params()
        params.time_stepping.t_end = 0.3
        params.init_fields.type = "noise"

        params.forcing.enable = True
        params.forcing.type = "in_script"

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.2

        periods.spatiotemporal_spectra = 0.01

        periods.spatial_means = 0.05
        periods.spatial_means_regions = 0.05

        for tag in params.output._tag_children:
            if tag.startswith("periods"):
                continue
            child = params.output[tag]
            if hasattr(child, "HAS_TO_PLOT_SAVED"):
                child["HAS_TO_PLOT_SAVED"] = True

        params.output.spectra.kzkh_periodicity = 2

        params.output.spatiotemporal_spectra.probes_region = (
            params.oper.nx // 2,
            params.oper.ny // 2,
            params.oper.nz // 2,
        )

    @pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
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
        sim.state.check_energy_equal_phys_spect()

        sim.state.compute("divh")

        np.testing.assert_almost_equal(
            sum(sim.output.compute_energies()), sim.output.compute_energy()
        )

        if mpi.nb_proc > 1:
            return

        sim2 = fls.load_sim_for_plot(sim.output.path_run)
        sim2.output.print_stdout.load()
        sim2.output.print_stdout.plot()
        sim2.output.print_stdout.plot_clock_times()

        data = sim2.output.spatial_means.load()
        assert "EKhs" in data
        sim2.output.spatial_means.load_dataset()
        sim2.output.spatial_means.plot(plot_injection=True, plot_hyper=True)
        sim2.output.spatial_means.plot_dt_E()

        sim2.output.spatial_means_regions.load()
        sim2.output.spatial_means_regions.plot()
        sim2.output.spatial_means_regions.plot_budget(
            iregion=1, decompose_fluxes=True, plot_conversion=True
        )

        sim2.output.spatial_means.plot_dimless_numbers_versus_time()
        result = sim2.output.spatial_means.get_dimless_numbers_averaged()
        assert isinstance(result, dict)

        sim2.output.spectra.load1d_mean()
        sim2.output.spectra.load3d_mean()
        sim2.output.spectra.load_kzkh_mean()
        sim2.output.spectra.plot1d()
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
        sim2.output.spectra.plot_kzkh(key="Khd")
        sim2.output.spectra.plot_kzkh_cumul_diss(tmin=0.1, tmax=10)

        sim2.output.spect_energy_budg.plot_kzkh()
        sim2.output.spect_energy_budg.plot_fluxes()

        sim2.output.phys_fields.set_equation_crosssection(f"x={sim.oper.Lx/4}")
        sim2.output.phys_fields.animate("vx")
        sim2.output.phys_fields.plot(field="vx", time=10)

        sim2.output.spatiotemporal_spectra.plot_kzkhomega(
            key_field="Kp", equation="omega=200 * N"
        )
        sim2.output.spatiotemporal_spectra.plot_temporal_spectra()

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
        sim.state.check_energy_equal_phys_spect()


class TestNoShearModes(TestSimulBase):
    @classmethod
    def init_params(self):
        params = super().init_params()

        params.init_fields.type = "noise"

        params.forcing.enable = False

        params.output.periods_save.spatial_means = 0.2
        params.output.periods_save.spectra = 0.2
        params.output.spectra.kzkh_periodicity = 1

        params.oper.NO_SHEAR_MODES = True
        params.no_vz_kz0 = True

        # we need delta_kx == delta_ky for this test
        params.oper.Ly = params.oper.Lx
        params.oper.ny = params.oper.nx

    def test_noshearmodes(self):

        sim = self.sim

        sim.time_stepping.start()
        sim.state.check_energy_equal_phys_spect()

        data = sim.output.spectra.load_kzkh_mean(
            tmin=0.2, key_to_load=["Khd", "Kz", "Khr", "A"]
        )
        means = sim.output.spatial_means.load()
        if mpi.nb_proc > 1:
            mpi.comm.barrier()

        # check k \cdot v = 0
        assert np.allclose(data["Khd"][0, :].sum(), 0.0)
        assert np.allclose(data["Kz"][:, 0].sum(), 0.0)

        # check energy in shear modes
        EKhs = means["EKhs"]
        E = means["E"]
        ratio = EKhs[-1] / E[-1]
        assert ratio < 1e-15

        # check energy in shear modes
        spectrum = data["Khr"] + data["Khd"] + data["Kz"] + data["A"]
        assert np.allclose(spectrum[:, 0].sum(), 0.0)

        # check energy in kz = 0
        spectrum = data["Khd"] + data["Kz"]
        assert np.allclose(spectrum[0, :].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
