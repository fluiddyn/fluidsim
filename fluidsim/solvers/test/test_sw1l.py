import unittest

from fluiddyn.util import mpi
from fluidsim.solvers.test.test_ns import TestSolver


class TestSW1L(TestSolver):
    solver = 'SW1L'
    options = {'HAS_TO_SAVE': True, 'forcing_enable': True}
    zero = 1e-7

    def _get_tendencies(self):
        self.sim.params.forcing.enable = False
        tendencies_fft = self.sim.tendencies_nonlin()

        Fx_fft = tendencies_fft.get_var('ux_fft')
        Fy_fft = tendencies_fft.get_var('uy_fft')
        Feta_fft = tendencies_fft.get_var('eta_fft')

        return Fx_fft, Fy_fft, Feta_fft

    def test_enstrophy_conservation(self):
        """Verify that the enstrophy growth rate due to nonlinear tendencies
        (advection term) must be zero.

        .. FIXME: Improve this test

        """
        self.sim.params.forcing.enable = False
        tendencies_fft = self.sim.tendencies_nonlin()

        state_spect = self.sim.state.state_spect
        oper = self.sim.oper
        try:
            Fq_fft = tendencies_fft.get_var('q_fft')
            q_fft = state_spect.get_var('q_fft')
        except ValueError:
            Fx_fft, Fy_fft, Feta_fft = self._get_tendencies()
            Fq_fft, Fap_fft, Fam_fft = oper.qapamfft_from_uxuyetafft(
                Fx_fft, Fy_fft, Feta_fft)
            q_fft = self.sim.state.get_var('q_fft')

        T_q = (Fq_fft.conj() * q_fft +
               Fq_fft * q_fft.conj()).real / 2.
        sum_T = oper.sum_wavenumbers(T_q)
        self.assertZero(sum_T, 5, warn=False)

    def test_energy_conservation(self):
        """Verify that the energy growth rate due to nonlinear tendencies
        (advection term) must be zero.

        """
        Fx_fft, Fy_fft, Feta_fft = self._get_tendencies()
        state_phys = self.sim.state.state_phys
        ux = state_phys.get_var('ux')
        uy = state_phys.get_var('uy')
        eta = state_phys.get_var('eta')

        oper = self.sim.oper
        Fx = oper.ifft2(Fx_fft)
        Fy = oper.ifft2(Fy_fft)
        Feta = oper.ifft2(Feta_fft)
        A = (Feta * (ux ** 2 + uy ** 2) / 2 +
             (1 + eta) * (ux * Fx + uy * Fy) +
             self.sim.params.c2 * eta * Feta)

        A_fft = oper.fft2(A)
        if mpi.rank == 0:
            self.assertZero(A_fft[0, 0])


class TestSW1LExactLin(TestSW1L):
    solver = 'SW1L.exactlin'
    options = {'HAS_TO_SAVE': True, 'forcing_enable': False}

    def _get_tendencies(self):
        self.sim.params.forcing.enable = False
        tendencies_fft = self.sim.tendencies_nonlin()

        Fap_fft = tendencies_fft.get_var('ap_fft')
        Fam_fft = tendencies_fft.get_var('am_fft')
        try:
            Fq_fft = tendencies_fft.get_var('q_fft')
        except ValueError:
            Fq_fft = self.sim.oper.create_arrayK(value=0.j)

        return self.sim.oper.uxuyetafft_from_qapamfft(Fq_fft, Fap_fft, Fam_fft)


class TestSW1LOnlyWaves(TestSW1LExactLin):
    solver = 'SW1L.onlywaves'
    options = {'HAS_TO_SAVE': True, 'forcing_enable': False}

    def test_enstrophy_conservation(self):
        # This solver does not update potential vorticity
        pass


class TestSW1LModify(TestSW1L):
    solver = 'SW1L.modified'
    options = {'HAS_TO_SAVE': True, 'forcing_enable': False}

    def test_enstrophy_conservation(self):
        # Theoretically not expected to conserve quadratic potential enstrophy
        pass

    def test_energy_conservation(self):
        """Verify that the energy growth rate due to nonlinear tendencies
        (advection term) must be zero.

        """
        Fx_fft, Fy_fft, Feta_fft = self._get_tendencies()
        state_phys = self.sim.state.state_phys
        ux = state_phys.get_var('ux')
        uy = state_phys.get_var('uy')
        eta = state_phys.get_var('eta')

        oper = self.sim.oper
        Fx = oper.ifft2(Fx_fft)
        Fy = oper.ifft2(Fy_fft)
        Feta = oper.ifft2(Feta_fft)
        A = ((ux * Fx + uy * Fy) + self.sim.params.c2 * eta * Feta)

        A_fft = oper.fft2(A)
        if mpi.rank == 0:
            self.assertZero(A_fft[0, 0])

    def test_energy_conservation_fft(self):
        """Verify that the quadratic energy growth rate due to nonlinear
        tendencies (advection term) must be zero in the spectral plane.

        """
        Fx_fft, Fy_fft, Feta_fft = self._get_tendencies()
        state = self.sim.state
        ux_fft = state.get_var('ux_fft')
        uy_fft = state.get_var('uy_fft')
        eta_fft = state.get_var('eta_fft')

        oper = self.sim.oper
        T_ux = (ux_fft.conj() * Fx_fft).real
        T_uy = (uy_fft.conj() * Fy_fft).real
        T_eta = (eta_fft.conj() * Feta_fft).real * self.sim.params.c2
        T_tot = T_ux + T_uy + T_eta
        sum_T = oper.sum_wavenumbers(T_tot)
        self.assertZero(sum_T)


class TestSW1LExmod(TestSW1LModify):
    solver = 'SW1L.exactlin.modified'
    options = {'HAS_TO_SAVE': True, 'forcing_enable': False}

    def _get_tendencies(self):
        self.sim.params.forcing.enable = False
        tendencies_fft = self.sim.tendencies_nonlin()

        Fq_fft = tendencies_fft.get_var('q_fft')
        Fap_fft = tendencies_fft.get_var('ap_fft')
        Fam_fft = tendencies_fft.get_var('am_fft')

        return self.sim.oper.uxuyetafft_from_qapamfft(Fq_fft, Fap_fft, Fam_fft)


if __name__ == '__main__':
    unittest.main()
