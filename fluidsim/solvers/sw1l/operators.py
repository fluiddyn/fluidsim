"""Operators sw1l (:mod:`fluidsim.operators.operators`)
=======================================================

Provides

.. autoclass:: OperatorsPseudoSpectralSW1L
   :members:
   :private-members:

"""

import numpy as np
from fluiddyn.util.compat import cached_property

from transonic import boost, jit, Array

from fluidsim.operators.operators2d import (
    OperatorsPseudoSpectral2D,
    rank,
    laplacian_fft,
    invlaplacian_fft,
)

AC = Array[np.complex128, "2d"]
AF = Array[np.float64, "2d"]


@jit
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


@boost
class OperatorsPseudoSpectralSW1L(OperatorsPseudoSpectral2D):
    Kappa_over_ic: AC
    nK0_loc: int
    nK1_loc: int
    rank: int

    @cached_property
    def Kappa2_not0(self):
        return self.K2_not0 + self.params.kd2

    @cached_property
    def Kappa_over_ic(self):
        Kappa2 = self.K2 + self.params.kd2
        return -1.0j * np.sqrt(Kappa2 / self.params.c2)

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

    def uxuyetafft_from_qapamfft(self, q_fft, ap_fft, am_fft):
        """q, ap, am (fft) ---> ux, uy, eta (fft)"""
        a_fft = ap_fft + am_fft
        if rank == 0:
            a_fft[0, 0] = 0.0
        div_fft = self.divfft_from_apamfft(ap_fft, am_fft)
        (uxa_fft, uya_fft, etaa_fft) = self.uxuyetafft_from_afft(a_fft)
        (uxq_fft, uyq_fft, etaq_fft) = self.uxuyetafft_from_qfft(q_fft)
        uxd_fft, uyd_fft = self.vecfft_from_divfft(div_fft)
        ux_fft = uxa_fft + uxq_fft + uxd_fft
        uy_fft = uya_fft + uyq_fft + uyd_fft
        eta_fft = etaa_fft + etaq_fft
        if rank == 0:
            ux_fft[0, 0] = 0.5 * (ap_fft[0, 0] + am_fft[0, 0])
            uy_fft[0, 0] = 0.5j * (am_fft[0, 0] - ap_fft[0, 0])
        return ux_fft, uy_fft, eta_fft

    def vecfft_from_rotdivfft(self, rot_fft, div_fft):
        """Inverse of the Helmholtz decomposition."""
        # TODO: Pythranize
        urx_fft, ury_fft = self.vecfft_from_rotfft(rot_fft)
        udx_fft, udy_fft = self.vecfft_from_divfft(div_fft)
        ux_fft = urx_fft + udx_fft
        uy_fft = ury_fft + udy_fft
        return ux_fft, uy_fft

    def uxuyetafft_from_qfft(self, q_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
            Kappa2_not0 = self.Kappa2_not0
        else:
            Kappa2_not0 = self.K2_not0 + params.kd2

        ilq_fft = invlaplacian_fft(q_fft, Kappa2_not0, rank)
        rot_fft = laplacian_fft(ilq_fft, self.K2)
        ux_fft, uy_fft = self.vecfft_from_rotfft(rot_fft)

        if params.f == 0:
            eta_fft = self.create_arrayK(value=0)
        else:
            eta_fft = -params.f * ilq_fft / params.c2
        return ux_fft, uy_fft, eta_fft

    def uxuyetafft_from_afft(self, a_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params

        eta_fft = self.etafft_from_afft(a_fft, params)
        if params.f == 0:
            rot_fft = self.create_arrayK(value=0)
        else:
            rot_fft = params.f * eta_fft

        ux_fft, uy_fft = self.vecfft_from_rotfft(rot_fft)

        return ux_fft, uy_fft, eta_fft

    def rotfft_from_qfft(self, q_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
            Kappa2_not0 = self.Kappa2_not0
        else:
            Kappa2_not0 = self.K2_not0 + params.kd2

        rot_fft = laplacian_fft(
            invlaplacian_fft(q_fft, Kappa2_not0, rank), self.K2
        )
        return rot_fft

    def rotfft_from_afft(self, a_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params

        if params.f == 0:
            rot_fft = self.create_arrayK(value=0)
        else:
            rot_fft = params.f * self.etafft_from_afft(a_fft, params)

        return rot_fft

    def afft_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft, params=None):
        if params is None:
            params = self.params

        a_fft = laplacian_fft(eta_fft, self.K2)
        if params.f != 0:
            rot_fft = self.rotfft_from_vecfft(ux_fft, uy_fft)
            a_fft += params.f / params.c2 * rot_fft
        return a_fft

    def etafft_from_qfft(self, q_fft, params=None):
        """Compute eta in Fourier space."""
        if params is None:
            params = self.params
            Kappa2_not0 = self.Kappa2_not0
        else:
            Kappa2_not0 = self.K2_not0 + params.kd2

        if params.f == 0:
            eta_fft = self.create_arrayK(value=0)
        else:
            eta_fft = (
                -params.f / params.c2 * invlaplacian_fft(q_fft, Kappa2_not0, rank)
            )
        return eta_fft

    def etafft_from_afft(self, a_fft, params=None):
        """Compute eta in Fourier space."""
        if params is None:
            params = self.params
            Kappa2_not0 = self.Kappa2_not0
        else:
            Kappa2_not0 = self.K2_not0 + params.kd2

        eta_fft = invlaplacian_fft(a_fft, Kappa2_not0, rank)
        return eta_fft

    def etafft_from_aqfft(self, a_fft, q_fft, params=None):
        """Compute eta in Fourier space."""
        if params is None:
            params = self.params
            Kappa2_not0 = self.Kappa2_not0
        else:
            Kappa2_not0 = self.K2_not0 + params.kd2

        if params.f == 0:
            eta_fft = invlaplacian_fft(a_fft, self.K2_not0, rank)
        else:
            eta_fft = invlaplacian_fft(
                (a_fft - params.f / params.c2 * q_fft), Kappa2_not0, rank
            )
        return eta_fft

    def qdafft_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft, params=None):
        if params is None:
            params = self.params
        div_fft = self.divfft_from_vecfft(ux_fft, uy_fft)
        rot_fft = self.rotfft_from_vecfft(ux_fft, uy_fft)
        q_fft = rot_fft - params.f * eta_fft
        ageo_fft = params.f / params.c2 * rot_fft + laplacian_fft(
            eta_fft, self.K2
        )
        return q_fft, div_fft, ageo_fft

    def apamfft_from_adfft(self, a_fft, d_fft):
        """Return the eigen modes ap and am."""
        Delta_a_fft = self.Kappa_over_ic * d_fft
        ap_fft = 0.5 * (a_fft + Delta_a_fft)
        am_fft = 0.5 * (a_fft - Delta_a_fft)
        return ap_fft, am_fft

    @jit
    def divfft_from_apamfft(self, ap_fft: AC, am_fft: AC):
        """Return div from the eigen modes ap and am."""
        n0 = self.nK0_loc
        n1 = self.nK1_loc
        Kappa_over_ic = self.Kappa_over_ic
        rank = self.rank

        Delta_a_fft = ap_fft - am_fft
        d_fft = np.empty([n0, n1], dtype=np.complex128)

        for i0 in range(n0):
            for i1 in range(n1):
                if i0 == 0 and i1 == 0 and rank == 0:
                    d_fft[i0, i1] = 0.0
                else:
                    d_fft[i0, i1] = Delta_a_fft[i0, i1] / Kappa_over_ic[i0, i1]
        return d_fft

    def qapamfft_from_etafft(self, eta_fft, params=None):
        """eta (fft) ---> q, ap, am (fft)"""
        if params is None:
            params = self.params

        q_fft = -params.f * eta_fft
        ap_fft = 0.5 * laplacian_fft(eta_fft, self.K2)
        am_fft = ap_fft.copy()
        return q_fft, ap_fft, am_fft
