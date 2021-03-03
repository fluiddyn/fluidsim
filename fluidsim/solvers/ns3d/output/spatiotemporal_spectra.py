"""
Spatiotemporal Spectra (:mod:`fluidsim.solvers.ns3d.output.spatiotemporal_spectra`)
===================================================================================

Provides:

.. autoclass:: SpatioTemporalSpectraNS3D
   :members:
   :private-members:

"""

from pathlib import Path

import numpy as np
import h5py

from fluidsim.base.output.spatiotemporal_spectra import SpatioTemporalSpectra


class SpatioTemporalSpectraNS3D(SpatioTemporalSpectra):
    def loop_spectra_kzkhomega(self, spectrum_k0k1k2omega, khs, KH, kzs, KZ):
        """Compute the kz-kh-omega spectrum."""
        deltakh = khs[1]
        deltakz = kzs[1]
        nkh = len(khs)
        nkz = len(kzs)
        nk0, nk1, nk2, nomega = spectrum_k0k1k2omega.shape
        spectrum_kzkhomega = np.zeros((nkz, nkh, nomega))
        for ik0 in range(nk0):
            for ik1 in range(nk1):
                for ik2 in range(nk2):
                    value = spectrum_k0k1k2omega[ik0, ik1, ik2, :]
                    kappa = KH[ik0, ik1, ik2]
                    ikh = int(kappa / deltakh)
                    kz = abs(KZ[ik0, ik1, ik2])
                    ikz = int(round(kz / deltakz))
                    if ikz >= nkz - 1:
                        ikz = nkz - 1
                    if ikh >= nkh - 1:
                        ikh = nkh - 1
                        spectrum_kzkhomega[ikz, ikh, :] += value
                    else:
                        coef_share = (kappa - khs[ikh]) / deltakh
                        spectrum_kzkhomega[ikz, ikh, :] += (
                            1 - coef_share
                        ) * value
                        spectrum_kzkhomega[ikz, ikh + 1, :] += coef_share * value
        return spectrum_kzkhomega

    def save_spectra_kzkhomega(self, tmin=0, tmax=None):
        """save the spatiotemporal spectra, with a cylindrical average in k-space"""
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        # compute spectra
        dict_spectra = self.compute_spectra(tmin=tmin, tmax=tmax)

        # get kz, kh
        oper = self.sim.oper
        order = dict_spectra["dims_order"]
        KZ = oper.deltakz * dict_spectra[f"K{order[0]}_adim"]
        KY = oper.deltaky * dict_spectra[f"K{order[1]}_adim"]
        KX = oper.deltaky * dict_spectra[f"K{order[1]}_adim"]
        KH = np.sqrt(KX ** 2 + KY ** 2)

        kz_spectra = np.arange(0, KZ.max() + 1e-15, oper.deltakz)

        deltakh = oper.deltakh
        khmax_spectra = min(KX.max(), KY.max())
        nkh_spectra = max(2, int(khmax_spectra / deltakh))
        kh_spectra = deltakh * np.arange(nkh_spectra)

        # perform cylindrical average
        dict_spectra_kzkhomega = {
            "kz_spectra": kz_spectra,
            "kh_spectra": kh_spectra,
            "omegas": dict_spectra["omegas"],
        }
        for key, data in dict_spectra.items():
            if not key.startswith("spect"):
                continue
            dict_spectra_kzkhomega[key] = self.loop_spectra_kzkhomega(
                data, kh_spectra, KH, kz_spectra, KZ
            )

        # save to file
        path_file = Path(self.sim.output.path_run) / "spatiotemporal_spectra.h5"
        with h5py.File(path_file, "w") as file:
            for key, val in dict_spectra_kzkhomega.items():
                file.create_dataset(key, data=val)

    def plot_kzkhomega(self, tmin=0, tmax=None):
        """plot the spatiotemporal spectra, with a cylindrical average in k-space"""
        if tmax is None:
            tmax = self.sim.params.time_stepping.t_end

        # load time
