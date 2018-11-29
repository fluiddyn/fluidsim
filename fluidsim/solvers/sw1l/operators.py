"""Operators sw1l (:mod:`fluidsim.operators.operators`)
=======================================================

Provides

.. autoclass:: OperatorsPseudoSpectralSW1L
   :members:
   :private-members:

"""

import numpy as np

# pythran import numpy as np

from fluidpythran import cachedjit, Array

from fluidsim.operators.operators2d import OperatorsPseudoSpectral2D, rank

AC = Array[np.complex128, "2d"]
AF = Array[np.float64, "2d"]


@cachedjit
def _qapamfft_from_uxuyetafft(
    ux_fft: AC,
    uy_fft: AC,
    eta_fft: AC,
    n0: int,
    n1: int,
    KX: AF,
    KY: AF,
    K2: AF,
    Kappa_over_ic: AC,
    f: float,
    c2: float,
    rank: int,
):
    """Calculate normal modes from primitive variables."""
    freq_Corio = f
    f_over_c2 = freq_Corio / c2

    q_fft = np.empty([n0, n1], dtype=np.complex128)
    ap_fft = np.empty([n0, n1], dtype=np.complex128)
    am_fft = np.empty([n0, n1], dtype=np.complex128)

    if freq_Corio != 0:
        for i0 in range(n0):
            for i1 in range(n1):
                if i0 == 0 and i1 == 0 and rank == 0:
                    q_fft[i0, i1] = 0
                    ap_fft[i0, i1] = ux_fft[0, 0] + 1.0j * uy_fft[0, 0]
                    am_fft[i0, i1] = ux_fft[0, 0] - 1.0j * uy_fft[0, 0]
                else:

                    rot_fft = 1j * (
                        KX[i0, i1] * uy_fft[i0, i1] - KY[i0, i1] * ux_fft[i0, i1]
                    )

                    q_fft[i0, i1] = rot_fft - freq_Corio * eta_fft[i0, i1]

                    a_over2_fft = 0.5 * (
                        K2[i0, i1] * eta_fft[i0, i1] + f_over_c2 * rot_fft
                    )

                    Deltaa_over2_fft = (
                        0.5j
                        * Kappa_over_ic[i0, i1]
                        * (
                            KX[i0, i1] * ux_fft[i0, i1]
                            + KY[i0, i1] * uy_fft[i0, i1]
                        )
                    )

                    ap_fft[i0, i1] = a_over2_fft + Deltaa_over2_fft
                    am_fft[i0, i1] = a_over2_fft - Deltaa_over2_fft

    else:  # (freq_Corio == 0.)
        for i0 in range(n0):
            for i1 in range(n1):
                if i0 == 0 and i1 == 0 and rank == 0:
                    q_fft[i0, i1] = 0
                    ap_fft[i0, i1] = ux_fft[0, 0] + 1.0j * uy_fft[0, 0]
                    am_fft[i0, i1] = ux_fft[0, 0] - 1.0j * uy_fft[0, 0]
                else:
                    q_fft[i0, i1] = 1j * (
                        KX[i0, i1] * uy_fft[i0, i1] - KY[i0, i1] * ux_fft[i0, i1]
                    )

                    a_over2_fft = 0.5 * K2[i0, i1] * eta_fft[i0, i1]

                    Deltaa_over2_fft = (
                        0.5j
                        * Kappa_over_ic[i0, i1]
                        * (
                            KX[i0, i1] * ux_fft[i0, i1]
                            + KY[i0, i1] * uy_fft[i0, i1]
                        )
                    )

                    ap_fft[i0, i1] = a_over2_fft + Deltaa_over2_fft
                    am_fft[i0, i1] = a_over2_fft - Deltaa_over2_fft

    return q_fft, ap_fft, am_fft


class OperatorsPseudoSpectralSW1L(OperatorsPseudoSpectral2D):
    def qapamfft_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft, params=None):
        """ux, uy, eta (fft) ---> q, ap, am (fft)"""

        if params is None:
            params = self.params

        n0 = self.nK0_loc
        n1 = self.nK1_loc

        KX = self.KX
        KY = self.KY
        K2 = self.K2
        Kappa_over_ic = self.Kappa_over_ic
        f = float(params.f)
        c2 = float(params.c2)

        return _qapamfft_from_uxuyetafft(
            ux_fft,
            uy_fft,
            eta_fft,
            n0,
            n1,
            KX,
            KY,
            K2,
            Kappa_over_ic,
            f,
            c2,
            rank,
        )

    def vecfft_from_rotdivfft(self, rot_fft, div_fft):
        """Inverse of the Helmholtz decomposition."""
        # TODO: Pythranize
        urx_fft, ury_fft = self.vecfft_from_rotfft(rot_fft)
        udx_fft, udy_fft = self.vecfft_from_divfft(div_fft)
        ux_fft = urx_fft + udx_fft
        uy_fft = ury_fft + udy_fft
        return ux_fft, uy_fft
