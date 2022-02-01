"""
Spatiotemporal Spectra (:mod:`fluidsim.solvers.ns3d.output.spatiotemporal_spectra`)
===================================================================================

Provides:

.. autoclass:: SpatioTemporalSpectraNS3D
   :members:
   :private-members:

"""

from math import pi

import numpy as np

from fluidsim.base.output.spatiotemporal_spectra import (
    SpatioTemporalSpectra3D,
    SpatioTemporalSpectraNS,
)

from transonic import boost, Array, Type

A4 = Array[Type(np.float32, np.float64), "4d", "C"]
A3 = "float[:,:,:]"
A1 = "float[:]"


@boost
def compute_spectrum_kzkhomega(
    field_k0k1k2omega: A4, khs: A1, kzs: A1, KX: A3, KZ: A3, KH: A3
):
    """Compute the kz-kh-omega spectrum."""
    deltakh = khs[1]
    deltakz = kzs[1]

    nkh = len(khs)
    nkz = len(kzs)
    nk0, nk1, nk2, nomega = field_k0k1k2omega.shape
    spectrum_kzkhomega = np.zeros(
        (nkz, nkh, nomega), dtype=field_k0k1k2omega.dtype
    )

    for ik0 in range(nk0):
        for ik1 in range(nk1):
            for ik2 in range(nk2):
                values = field_k0k1k2omega[ik0, ik1, ik2, :]
                kx = KX[ik0, ik1, ik2]
                if kx != 0.0:
                    # warning: we should also consider another condition
                    # (kx != kx_max) but it is not necessary here mainly
                    # because of dealiasing
                    values = 2 * values

                kappa = KH[ik0, ik1, ik2]
                ikh = int(kappa / deltakh)
                kz = abs(KZ[ik0, ik1, ik2])
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
                        spectrum_kzkhomega[ikz, ikh, i] += (
                            1 - coef_share
                        ) * value
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


def _sum_wavenumber3D(field, KX, kx_max):
    n0, n1, n2 = field.shape[:3]
    result = 0.0
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                value = field[i0, i1, i2]
                kx = KX[i0, i1, i2]
                if kx != 0.0 and kx != kx_max:
                    value *= 2
                result += value
    return result


class SpatioTemporalSpectraNS3D(SpatioTemporalSpectraNS, SpatioTemporalSpectra3D):

    compute_spectrum_kzkhomega = staticmethod(compute_spectrum_kzkhomega)
    _sum_wavenumber = staticmethod(_sum_wavenumber3D)

    def compute_spectra_urud(self, tmin=0, tmax=None, dtype=None):
        """compute the spectra of ur, ud from files"""
        # load time series as state_spect arrays + times
        series = self.load_time_series(
            keys=("vx", "vy"), tmin=tmin, tmax=tmax, dtype=dtype
        )

        # toroidal/poloidal decomposition
        # urx_fft, ury_fft contain shear modes!
        vx_fft = series["vx_Fourier"]
        vy_fft = series["vy_Fourier"]
        if vx_fft.dtype == "complex64":
            float_dtype = "float32"
        elif vx_fft.dtype == "complex128":
            float_dtype = "float64"

        params_oper = self.sim.params.oper
        deltaky = 2 * pi / params_oper.Ly
        deltakx = 2 * pi / params_oper.Lx

        order = series["dims_order"]

        shapeK = series[f"K{order[1]}_adim"].shape
        KY = np.zeros(shapeK + (1,), dtype=float_dtype)
        KX = np.zeros(shapeK + (1,), dtype=float_dtype)
        KY[..., 0] = deltaky * series[f"K{order[1]}_adim"]
        KX[..., 0] = deltakx * series[f"K{order[2]}_adim"]

        inv_Kh_square_nozero = KX**2 + KY**2
        inv_Kh_square_nozero[inv_Kh_square_nozero == 0] = 1e-14
        inv_Kh_square_nozero = 1 / inv_Kh_square_nozero

        kdotu_fft = KX * vx_fft + KY * vy_fft
        udx_fft = kdotu_fft * KX * inv_Kh_square_nozero
        udy_fft = kdotu_fft * KY * inv_Kh_square_nozero

        urx_fft = vx_fft - udx_fft
        ury_fft = vy_fft - udy_fft

        del vx_fft, vy_fft, KX, KY, inv_Kh_square_nozero, kdotu_fft

        # perform time fft
        print("computing temporal spectra...")

        spectra = {k: v for k, v in series.items() if k.startswith("K")}
        del series

        # ud
        spectra["spectrum_Khd"] = Khd = np.zeros(udx_fft.shape, dtype=dtype)
        freq, spectrum = self._compute_spectrum(udx_fft)
        Khd += 0.5 * spectrum
        freq, spectrum = self._compute_spectrum(udy_fft)
        Khd += 0.5 * spectrum

        # ur
        spectra["spectrum_Khr"] = Khr = np.zeros(udx_fft.shape, dtype=dtype)
        freq, spectrum = self._compute_spectrum(urx_fft)
        Khr += 0.5 * spectrum
        freq, spectrum = self._compute_spectrum(ury_fft)
        Khr += 0.5 * spectrum

        spectra["omegas"] = 2 * pi * freq
        spectra["dims_order"] = order

        return spectra
