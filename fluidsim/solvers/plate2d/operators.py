"""Operators plate2d model (:mod:`fluidsim.operators.operators_plate2d`)
========================================================================

Provides

.. autoclass:: OperatorsPseudoSpectralPlate2D
   :members:
   :private-members:

"""

import numpy as np

from transonic import jit, Array

from ...operators.operators2d import OperatorsPseudoSpectral2D

AC2 = Array[np.complex128, "2d"]
AF2 = Array[np.float64, "2d"]


@jit
def monge_ampere_step0(a_fft: AC2, b_fft: AC2, KX2: AF2, KY2: AF2, KXKZ: AF2):
    pxx_a_fft = -a_fft * KX2
    pyy_a_fft = -a_fft * KY2
    pxy_a_fft = -a_fft * KXKZ
    pxx_b_fft = -b_fft * KX2
    pyy_b_fft = -b_fft * KY2
    pxy_b_fft = -b_fft * KXKZ
    return pxx_a_fft, pyy_a_fft, pxy_a_fft, pxx_b_fft, pyy_b_fft, pxy_b_fft


@jit
def monge_ampere_step1(
    pxx_a: AF2, pyy_a: AF2, pxy_a: AF2, pxx_b: AF2, pyy_b: AF2, pxy_b: AF2
):
    return pxx_a * pyy_b + pyy_a * pxx_b - 2 * pxy_a * pxy_b


class OperatorsPseudoSpectralPlate2D(OperatorsPseudoSpectral2D):
    """Operators for the plate2d model."""

    def __init__(self, params):
        super().__init__(params)

        self.KX2 = self.KX**2
        self.KY2 = self.KY**2
        self.KXKY = self.KX * self.KY

        self.tmp_pxx_a = np.empty_like(self.XX)
        self.tmp_pyy_a = np.empty_like(self.XX)
        self.tmp_pxy_a = np.empty_like(self.XX)

        self.tmp_pxx_b = np.empty_like(self.XX)
        self.tmp_pyy_b = np.empty_like(self.XX)
        self.tmp_pxy_b = np.empty_like(self.XX)

    def monge_ampere_from_fft(self, a_fft, b_fft):
        """Compute the Monge-Ampere operator"""

        (
            pxx_a_fft,
            pyy_a_fft,
            pxy_a_fft,
            pxx_b_fft,
            pyy_b_fft,
            pxy_b_fft,
        ) = monge_ampere_step0(a_fft, b_fft, self.KX2, self.KY2, self.KXKY)

        self.ifft_as_arg(pxx_a_fft, self.tmp_pxx_a)
        self.ifft_as_arg(pyy_a_fft, self.tmp_pyy_a)
        self.ifft_as_arg(pxy_a_fft, self.tmp_pxy_a)
        self.ifft_as_arg(pxx_b_fft, self.tmp_pxx_b)
        self.ifft_as_arg(pyy_b_fft, self.tmp_pyy_b)
        self.ifft_as_arg(pxy_b_fft, self.tmp_pxy_b)

        return monge_ampere_step1(
            self.tmp_pxx_a,
            self.tmp_pyy_a,
            self.tmp_pxy_a,
            self.tmp_pxx_b,
            self.tmp_pyy_b,
            self.tmp_pxy_b,
        )


# def monge_ampere_from_fft_numpy(a_fft, b_fft, KX, KY, ifft2):

#     pxx_a = -ifft2(a_fft * KX ** 2)
#     pyy_a = -ifft2(a_fft * KY ** 2)
#     pxy_a = -ifft2(a_fft * KX * KY)

#     pxx_b = -ifft2(b_fft * KX ** 2)
#     pyy_b = -ifft2(b_fft * KY ** 2)
#     pxy_b = -ifft2(b_fft * KX * KY)

#     return pxx_a * pyy_b + pyy_a * pxx_b - 2 * pxy_a * pxy_b
