"""Stress tensor 3D (:mod:`fluidsim.base.turb_model.stress_tensor`)
===================================================================

Provides:

.. autoclass:: StressTensorComputer3D
   :members:
   :private-members:
   :noindex:
   :undoc-members:

"""

import numpy as np


class StressTensorComputer3D:
    def __init__(self, oper):
        self.oper = oper

    def grad_from_arr_fft(self, arr_fft):
        dx_arr_fft, dy_arr_fft, dz_arr_fft = self.oper.grad_fft_from_arr_fft(
            arr_fft
        )
        ifft = self.oper.ifft
        return ifft(dx_arr_fft), ifft(dy_arr_fft), ifft(dz_arr_fft)

    def compute_stress_tensor(self, ux_fft, uy_fft, uz_fft):

        dx_ux, dy_ux, dz_ux = self.grad_from_arr_fft(ux_fft)
        dx_uy, dy_uy, dz_uy = self.grad_from_arr_fft(uy_fft)
        dx_uz, dy_uz, dz_uz = self.grad_from_arr_fft(uz_fft)

        Sxx = dx_ux
        Syy = dy_uy
        Szz = dz_uz
        Syx = 0.5 * (dy_ux + dx_uy)
        Szx = 0.5 * (dz_ux + dx_uz)
        Szy = 0.5 * (dz_uy + dy_uz)

        return Sxx, Syy, Szz, Syx, Szx, Szy

    def compute_norm(self, Sxx, Syy, Szz, Syx, Szx, Szy):
        return np.sqrt(
            Sxx**2 + Syy**2 + Szz**2 + 2 * (Syx**2 + Szx**2 + Szy**2)
        )
