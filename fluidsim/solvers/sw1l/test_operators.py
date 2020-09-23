import unittest
import numpy as np
import sys

import fluiddyn.util.mpi as mpi
from fluiddyn.util.paramcontainer import ParamContainer

from fluidsim.util.testing import TestCase, stdout_redirected, skip_if_no_fluidfft


def create_oper(type_fft=None, coef_dealiasing=2.0 / 3):

    params = ParamContainer(tag="params")

    params._set_attrib("ONLY_COARSE_OPER", False)
    params._set_attrib("f", 0)
    params._set_attrib("c2", 100)
    params._set_attrib("kd2", 0)

    from fluidsim.solvers.sw1l.operators import OperatorsPseudoSpectralSW1L

    OperatorsPseudoSpectralSW1L._complete_params_with_default(params)

    if mpi.nb_proc == 1:
        nh = 9
    else:
        nh = 8

    params.oper.nx = nh
    params.oper.ny = nh
    Lh = 6.0
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    if type_fft is not None:
        params.oper.type_fft = type_fft

    params.oper.coef_dealiasing = coef_dealiasing

    oper = OperatorsPseudoSpectralSW1L(params=params)

    return oper


@skip_if_no_fluidfft
@unittest.skipIf(sys.platform.startswith("win"), "Untested on Windows")
class TestOperators(TestCase):
    @classmethod
    def setUpClass(cls):
        with stdout_redirected(cls.has_to_redirect_stdout):
            cls.oper = create_oper()
        cls.rtol = 1e-15
        cls.atol = 1e-14  # Absolute tolerance for double precision FFT

    def test_uxuyeta_qapam_conversion(self):
        """Test conversion back and forth from q,ap,am -> ux, uy, eta"""
        oper = self.oper
        q_fft = oper.create_arrayK_random()
        ap_fft = oper.create_arrayK_random()
        am_fft = oper.create_arrayK_random()
        if mpi.rank == 0:
            q_fft[0, 0] = ap_fft[0, 0] = am_fft[0, 0] = 0.0

        ux_fft, uy_fft, eta_fft = oper.uxuyetafft_from_qapamfft(
            q_fft, ap_fft, am_fft
        )
        q2_fft, ap2_fft, am2_fft = oper.qapamfft_from_uxuyetafft(
            ux_fft, uy_fft, eta_fft
        )

        np.testing.assert_allclose(q2_fft, q_fft, self.rtol, self.atol)
        np.testing.assert_allclose(ap2_fft, ap_fft, self.rtol, self.atol)
        np.testing.assert_allclose(am2_fft, am_fft, self.rtol, self.atol)


if __name__ == "__main__":
    unittest.main()
