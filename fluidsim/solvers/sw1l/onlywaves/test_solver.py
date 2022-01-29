import unittest
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
import fluiddyn.util.mpi as mpi
from fluidsim.util.testing import (
    TestSimulConserve,
    skip_if_no_fluidfft,
    classproperty,
)


@skip_if_no_fluidfft
class TestSimulSW1LWaves(TestSimulConserve):
    zero = 1e-6

    @classproperty
    def Simul(cls):
        from fluidsim.solvers.sw1l.onlywaves.solver import Simul

        return Simul

    @classmethod
    def init_params(cls):
        super().init_params()
        params = cls.params
        params.output.HAS_TO_SAVE = True

        params.output.periods_save.spect_energy_budg = 0.2
        params.init_fields.type = "noise"
        params.oper.nx = 16
        params.oper.ny = 8

    def get_tendencies(self):
        tendencies_fft = self.tendencies_fft

        Fq_fft = self.sim.oper.create_arrayK(value=0.0j)
        Fap_fft = tendencies_fft.get_var("ap_fft")
        Fam_fft = tendencies_fft.get_var("am_fft")

        return self.sim.oper.uxuyetafft_from_qapamfft(Fq_fft, Fap_fft, Fam_fft)

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

    def test_init_methods(self):
        compute = self.sim.state.compute
        ux_fft, uy_fft, eta_fft = map(compute, ("ux_fft", "uy_fft", "eta_fft"))
        self.sim.state.init_from_uxuyetafft(ux_fft, uy_fft, eta_fft)
        self.sim.state.init_from_etafft(eta_fft)

    def test_state_compute(self):
        for key in ("uy_fft", "rot", "q"):
            var_computed = self.sim.state.compute(key)


if __name__ == "__main__":
    unittest.main()
