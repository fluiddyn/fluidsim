import unittest

import fluiddyn.util.mpi as mpi
from fluidsim.util.testing import (
    TestSimulConserve,
    classproperty,
    skip_if_no_fluidfft,
)


@skip_if_no_fluidfft
class TestSimulSW1LExactlinModified(TestSimulConserve):
    @classproperty
    def Simul(cls):
        from fluidsim.solvers.sw1l.exactlin.modified.solver import Simul

        return Simul

    @classmethod
    def init_params(cls):
        super().init_params()
        params = cls.params
        params.output.HAS_TO_SAVE = True

        params.output.periods_save.spect_energy_budg = 0.2

        params.forcing.enable = True
        params.forcing.type = "potential"
        params.forcing.nkmin_forcing = 2
        params.forcing.nkmax_forcing = 4
        params.init_fields.type = "vortex_grid"
        params.oper.nx = 16
        params.oper.ny = 8

    def get_tendencies(self):
        tendencies_fft = self.tendencies_fft

        Fq_fft = tendencies_fft.get_var("q_fft")
        Fap_fft = tendencies_fft.get_var("ap_fft")
        Fam_fft = tendencies_fft.get_var("am_fft")

        return self.sim.oper.uxuyetafft_from_qapamfft(Fq_fft, Fap_fft, Fam_fft)

    # def test_enstrophy_conservation(self):
    #     """Theoretically not expected to conserve quadratic potential enstrophy"""
    #     pass

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
        A = (ux * Fx + uy * Fy) + self.sim.params.c2 * eta * Feta

        A_fft = oper.fft2(A)
        if mpi.rank == 0:
            self.assertAlmostZero(A_fft[0, 0])

    def test_energy_conservation_fft(self):
        """Verify that the quadratic energy growth rate due to nonlinear
        tendencies (advection term) must be zero in the spectral plane.

        """
        Fx_fft, Fy_fft, Feta_fft = self.get_tendencies()
        state = self.sim.state
        ux_fft = state.get_var("ux_fft")
        uy_fft = state.get_var("uy_fft")
        eta_fft = state.get_var("eta_fft")

        oper = self.sim.oper
        T_ux = (ux_fft.conj() * Fx_fft).real
        T_uy = (uy_fft.conj() * Fy_fft).real
        T_eta = (eta_fft.conj() * Feta_fft).real * self.sim.params.c2
        T_tot = T_ux + T_uy + T_eta
        sum_T = oper.sum_wavenumbers(T_tot)
        self.assertAlmostZero(sum_T)


if __name__ == "__main__":
    unittest.main()
