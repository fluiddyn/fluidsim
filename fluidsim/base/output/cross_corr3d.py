"""Cross correlations
=====================

Provides:

.. autoclass:: CrossCorrelations
   :members:
   :private-members:
   :noindex:
   :undoc-members:

"""


import itertools

import numpy as np

from fluidsim.base.output.spectra3d import BaseSpectra


def _make_small_key(key):
    small_key = key[:-4]
    if len(small_key) == 2 and small_key.startswith("v"):
        return small_key[1:]
    return small_key


class CrossCorrelations(BaseSpectra):

    _tag = "cross_corr"

    def compute(self):
        """compute the values at one time."""

        keys = self.sim.state.keys_state_spect

        combinations = itertools.combinations(keys, 2)

        dict_spectra1d = {}
        dict_spectra3d = {}

        has_to_save_kzkh = self.has_to_save_kzkh()

        if has_to_save_kzkh:
            dict_kzkh = {}
        else:
            dict_kzkh = None

        for key0, key1 in combinations:
            key_cc = _make_small_key(key0) + _make_small_key(key1)

            field0 = self.sim.state.get_var(key0)
            field1 = self.sim.state.get_var(key1)

            cross_corr_tmp = -np.real(field0.conj() * field1)

            cc_kx, cc_ky, cc_kz = self.oper.compute_1dspectra(cross_corr_tmp)

            dict_spectra1d[key_cc + "_kx"] = cc_kx
            dict_spectra1d[key_cc + "_ky"] = cc_ky
            dict_spectra1d[key_cc + "_kz"] = cc_kz
            dict_spectra3d[key_cc] = self.oper.compute_3dspectrum(cross_corr_tmp)

            if has_to_save_kzkh:
                dict_kzkh[key_cc] = self.oper.compute_spectrum_kzkh(
                    cross_corr_tmp
                )

        dict_spectra1d = {"spectra_" + k: v for k, v in dict_spectra1d.items()}
        dict_spectra3d = {"spectra_" + k: v for k, v in dict_spectra3d.items()}
        return dict_spectra1d, dict_spectra3d, dict_kzkh
