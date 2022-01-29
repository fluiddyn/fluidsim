import unittest

import matplotlib.pyplot as plt
from numpy.testing import assert_array_almost_equal

import fluiddyn.util.mpi as mpi
from fluidsim.util.testing import (
    TestSimulConserveOutput,
    classproperty,
    skip_if_no_fluidfft,
)


@skip_if_no_fluidfft
class TestSimulSW1L(TestSimulConserveOutput):
    zero = 1e-5

    @classproperty
    def Simul(cls):
        from fluidsim.solvers.sw1l.solver import Simul

        return Simul

    @classmethod
    def init_params(cls):
        super().init_params()
        params = cls.params
        params.output.HAS_TO_SAVE = True
        params.output.ONLINE_PLOT_OK = True
        params.output.spatial_means.HAS_TO_PLOT_SAVED = True
        params.output.spect_energy_budg.HAS_TO_PLOT_SAVED = True
        params.output.spectra.HAS_TO_PLOT_SAVED = True
        params.output.increments.HAS_TO_PLOT_SAVED = True
        params.output.pdf.HAS_TO_PLOT_SAVED = True

        params.output.periods_save.spatial_means = 0.25
        params.output.periods_save.phys_fields = 0.25
        params.output.periods_save.spect_energy_budg = 0.2
        params.output.periods_save.spectra = 0.2
        params.output.periods_save.increments = 0.25
        params.output.periods_save.pdf = 0.25
        params.output.periods_save.time_signals_fft = 0.1

        params.f = 1e-4
        params.forcing.enable = True
        params.forcing.type = "waves"
        params.forcing.nkmin_forcing = 2
        params.forcing.nkmax_forcing = 4
        params.init_fields.type = "wave"
        params.oper.nx = 16
        params.oper.ny = 16

    def test_phys_fields(self):
        self.plot("phys_fields")

    def test_print_stdout(self):
        self.plot("print_stdout")

    def test_spatial_means(self):
        self.plot("spatial_means")
        spatial_means = self.sim.output.spatial_means
        spatial_means.compute_time_means()
        spatial_means.time_first_saved()
        spatial_means.time_last_saved()

    def test_increments(self):
        self.plot("increments")
        self.sim.output.increments.plot_Kolmo()

    def test_spectra(self):
        """Test spectra loading and plotting.

        .. TODO::

           Errors

           * if (nx, ny) = (16, 8)
             >>> E_K = dset_spectrum1Dkx_EK[it] + dset_spectrum1Dky_EK[it]
             E ValueError: operands could not be broadcast together with
             shapes (9,) (5,)
           * if periods_save = 0.25.
             >>> delta_i_plot = int(np.round(delta_t / delta_t_save))
             E ValueError: cannot convert float NaN to integer

        """
        self.get_results("spectra")
        if mpi.nb_proc == 1:
            spectra = self.sim.output.spectra
            spectra.plot1d()
            spectra.plot2d()
            # spectra.plot_diss()
            spectra.compute_lin_spectra()
            plt.cla()

    def test_spect_energy_budg(self):
        """Test spect_energy_budg loading and plotting.

        .. TODO::

           Errors

           * if periods_save = 0.25.
             >>> delta_i_plot = int(np.round(delta_t / delta_t_save))
             E ValueError: cannot convert float NaN to integer

        """
        self.get_results("spect_energy_budg")
        self.plot("spect_energy_budg")

    def test_time_signals_fft(self):
        self.plot("time_signals_fft")
        self.sim.output.time_signals_fft.plot_spectra()

    def test_pdf(self):
        self.plot("pdf")
        self.get_results("pdf")

    def test_energy_vs_spatial_means(self):
        """Verify energy saved by spatial_means module is the same."""
        if mpi.nb_proc > 1:
            mpi.comm.barrier()
        dict_results_print_stdout = self.get_results("print_stdout")
        df_spatial_means = self.get_results("spatial_means")

        # ignore last row to be comparable to print_stdout
        imax = -1 if len(df_spatial_means) > 1 else None
        assert_array_almost_equal(
            dict_results_print_stdout["E"],
            df_spatial_means.E[:imax].values,
            decimal=4,
        )

    def get_tendencies(self):
        tendencies_fft = self.tendencies_fft
        Fx_fft = tendencies_fft.get_var("ux_fft")
        Fy_fft = tendencies_fft.get_var("uy_fft")
        Feta_fft = tendencies_fft.get_var("eta_fft")

        return Fx_fft, Fy_fft, Feta_fft

    def test_enstrophy_conservation(self):
        """Verify that the enstrophy growth rate due to nonlinear tendencies
        (advection term) must be zero.

        """
        oper = self.sim.oper
        Fx_fft, Fy_fft, Feta_fft = self.get_tendencies()
        Fq_fft, _, _ = oper.qapamfft_from_uxuyetafft(Fx_fft, Fy_fft, Feta_fft)
        q_fft = self.sim.state.get_var("q_fft")

        T_q = (Fq_fft.conj() * q_fft + Fq_fft * q_fft.conj()).real / 2.0
        sum_T = oper.sum_wavenumbers(T_q)
        self.assertAlmostZero(sum_T, tolerance_warning=False)

    def test_energy_conservation(self):
        """Verify that the energy growth rate due to nonlinear tendencies
        (advection term) must be zero.

        """
        Fx_fft, Fy_fft, Feta_fft = self.get_tendencies()
        state_phys = self.sim.state.state_phys
        ux = state_phys.get_var("ux")
        uy = state_phys.get_var("uy")
        eta = state_phys.get_var("eta")

        oper = self.sim.oper
        Fx = oper.ifft2(Fx_fft)
        Fy = oper.ifft2(Fy_fft)
        Feta = oper.ifft2(Feta_fft)
        A = (
            Feta * (ux**2 + uy**2) / 2
            + (1 + eta) * (ux * Fx + uy * Fy)
            + self.sim.params.c2 * eta * Feta
        )

        A_fft = oper.fft2(A)
        if mpi.rank == 0:
            self.assertAlmostZero(A_fft[0, 0], tolerance_warning=False)

    def test_state_methods(self):
        state_spect = self.sim.state.state_spect.copy()
        self.sim.state.statespect_from_statephys()
        state_spect2 = self.sim.state.state_spect
        assert id(state_spect) != id(state_spect2)
        assert_array_almost_equal(state_spect, state_spect2)

    def test_state_init_from_uxuyfft(self):
        get_var = self.sim.state.get_var
        ux_fft = get_var("ux_fft")
        uy_fft = get_var("uy_fft")

        # The following method assumes that the velocity only has a rotational
        # component
        self.sim.state.init_from_uxuyfft(ux_fft, uy_fft)

        # Should have no divergence
        div = self.sim.state.compute("div")
        self.assertAlmostZero(div.max())

    def test_state_compute(self):
        for key in ("q", "h", "Floc"):
            var_computed = self.sim.state.compute(key)


if __name__ == "__main__":
    unittest.main()
