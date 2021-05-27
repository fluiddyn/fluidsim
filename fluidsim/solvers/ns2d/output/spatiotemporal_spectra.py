"""
Spatiotemporal Spectra (:mod:`fluidsim.solvers.ns2d.output.spatiotemporal_spectra`)
===================================================================================

Provides:

.. autoclass:: SpatioTemporalSpectraNS2D
   :members:
   :private-members:

"""

import numpy as np

from fluidsim.base.output.spatiotemporal_spectra import (
    SpatioTemporalSpectra2D,
    SpatioTemporalSpectraNS,
)

from transonic import boost, Array, Type

A3 = Array[Type(np.float32, np.float64), "3d", "C"]
A2 = Array[np.float64, "2d", "C"]
A1 = "float[:]"


@boost
def compute_spectrum_kzkhomega(
    field_k0k1omega: A3, khs: A1, kzs: A1, KX: A2, KZ: A2, KH: A2
):
    """Compute the kz-kh-omega spectrum."""
    deltakh = khs[1]
    deltakz = kzs[1]

    nkh = len(khs)
    nkz = len(kzs)
    nk0, nk1, nomega = field_k0k1omega.shape
    spectrum_kzkhomega = np.zeros((nkz, nkh, nomega), dtype=field_k0k1omega.dtype)

    for ik0 in range(nk0):
        for ik1 in range(nk1):
            values = field_k0k1omega[ik0, ik1, :]
            kx = KX[ik0, ik1]
            if kx != 0.0:
                # warning: we should also consider another condition
                # (kx != kx_max) but it is not necessary here mainly
                # because of dealiasing
                values = 2 * values

            kappa = KH[ik0, ik1]
            ikh = int(kappa / deltakh)
            kz = abs(KZ[ik0, ik1])
            ikz = int(round(kz / deltakz))
            if ikz >= nkz - 1:
                ikz = nkz - 1
            if ikh >= nkh - 1:
                ikh = nkh - 1
                for i, value in enumerate(values):
                    spectrum_kzkhomega[ikz, ikh, i] += value
            else:
                coef_share = (kappa - khs[ikh]) / deltakh
                for i, value in enumerate(values):
                    spectrum_kzkhomega[ikz, ikh, i] += (1 - coef_share) * value
                    spectrum_kzkhomega[ikz, ikh + 1, i] += coef_share * value

    # get one-sided spectrum in the omega dimension
    nomega = (nomega + 1) // 2
    spectrum_onesided = np.zeros((nkz, nkh, nomega))
    spectrum_onesided[:, :, 0] = spectrum_kzkhomega[:, :, 0]
    spectrum_onesided[:, :, 1:] = (
        spectrum_kzkhomega[:, :, 1:nomega]
        + spectrum_kzkhomega[:, :, -1:-nomega:-1]
    )
    return spectrum_onesided / (deltakz * deltakh)


def _sum_wavenumber2D(field, KX, kx_max):
    n0, n1 = field.shape[:2]
    result = 0.0
    for i0 in range(n0):
        for i1 in range(n1):
            value = field[i0, i1]
            kx = KX[i0, i1]
            if kx != 0.0 and kx != kx_max:
                value *= 2
            result += value
    return result


class SpatioTemporalSpectraNS2D(SpatioTemporalSpectraNS, SpatioTemporalSpectra2D):

    compute_spectrum_kzkhomega = staticmethod(compute_spectrum_kzkhomega)
    _sum_wavenumber = staticmethod(_sum_wavenumber2D)

    def save_spectra_kzkhomega(self, tmin=0, tmax=None, dtype=None):
        return super().save_spectra_kzkhomega(tmin, tmax, dtype)
