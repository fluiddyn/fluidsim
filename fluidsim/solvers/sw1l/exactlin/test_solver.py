import unittest

import fluiddyn.util.mpi as mpi
from fluidsim.util.testing import (
    TestSimulConserveOutput,
    classproperty,
    skip_if_no_fluidfft,
)


@skip_if_no_fluidfft
class TestSimulSW1LExactlin(TestSimulConserveOutput):
    zero = 1e-3

    @classproperty
    def Simul(cls):
        from fluidsim.solvers.sw1l.exactlin.solver import Simul

        return Simul

    @classmethod
    def init_params(cls):
        super().init_params()
        params = cls.params
        params.output.HAS_TO_SAVE = True

        params.output.periods_save.spect_energy_budg = 0.2

        params.forcing.enable = True
        params.forcing.type = "waves_vortices"
        params.forcing.forcing_rate = 0.01
        params.forcing.nkmin_forcing = 2
        params.forcing.nkmax_forcing = 4
        params.init_fields.type = "noise"
        params.oper.nx = 16
        params.oper.ny = 8

    def get_tendencies(self):
        tendencies_fft = self.tendencies_fft

        Fq_fft = tendencies_fft.get_var("q_fft")
        Fap_fft = tendencies_fft.get_var("ap_fft")
        Fam_fft = tendencies_fft.get_var("am_fft")

        return self.sim.oper.uxuyetafft_from_qapamfft(Fq_fft, Fap_fft, Fam_fft)

    def test_enstrophy_conservation(self):
        """Verify that the enstrophy growth rate due to nonlinear tendencies
        (advection term) must be zero.

        .. FIXME: Improve this test

        """
        tendencies_fft = self.tendencies_fft
        return
        state_spect = self.sim.state.state_spect
        oper = self.sim.oper
        Fq_fft = tendencies_fft.get_var("q_fft")
        q_fft = state_spect.get_var("q_fft")

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


if __name__ == "__main__":
    unittest.main()
