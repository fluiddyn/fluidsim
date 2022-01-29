"""Normal mode decomposition for SW1L solvers
(:mod:`fluidsim.solvers.sw1l.output.normal_mode`)
=================================================

Provides:

.. autoclass:: NormalModeBase
   :members:
   :private-members:

.. autoclass:: NormalModeDecomposition
   :members:
   :private-members:

.. autoclass:: NormalModeDecompositionModified
   :members:
   :private-members:

"""

import numpy as np

from fluiddyn.util.compat import cached_property
from fluiddyn.util import mpi


class NormalModeBase:
    def __init__(self, output):
        self.sim = output.sim
        self.params = output.sim.params
        self.oper = output.oper

        f = self.params.f
        c = self.params.c2**0.5
        ck = c * self.oper.K_not0

        if f == 0:
            self.sigma = ck
        else:
            self.sigma = np.sqrt(f**2 + ck**2)

        self.bvec_fft = None
        self.it_bvec_fft_computed = -1

    def compute(self):
        if self.it_bvec_fft_computed != self.sim.time_stepping.it:
            get_var = self.sim.state.get_var
            if {"ap_fft", "am_fft"}.issubset(self.sim.state.keys_state_spect):
                q_fft = get_var("q_fft")
                ap_fft = get_var("ap_fft")
                am_fft = get_var("am_fft")
                bvec_fft = self.bvecfft_from_qapamfft(q_fft, ap_fft, am_fft)
            else:
                ux_fft = get_var("ux_fft")
                uy_fft = get_var("uy_fft")
                eta_fft = get_var("eta_fft")
                bvec_fft = self.bvecfft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

            self.bvec_fft = bvec_fft
            self.it_bvec_fft_computed = self.sim.time_stepping.it

        return self.bvec_fft

    def bvecfft_from_qapamfft(self, q_fft, ap_fft, am_fft):
        r"""Compute normal mode vector :math:`\mathbf{B}`
        with dimensions of velocity from diagonalized linear modes."""
        c = self.params.c2**0.5
        c2 = self.params.c2
        K = self.oper.K_not0
        sigma = self.sigma

        q_fft = -q_fft * c / sigma
        ap_fft = ap_fft * 2**0.5 * c2 / (sigma * K)
        am_fft = am_fft * 2**0.5 * c2 / (sigma * K)
        bvec_fft = np.array([q_fft, ap_fft, am_fft])
        if mpi.rank == 0 or self.oper.is_sequential:
            bvec_fft[:, 0, 0] = 0.0

        return bvec_fft

    def bvecfft_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft):
        r"""Compute normal mode vector, :math:`\mathbf{B}`
        with dimensions of velocity from primitive variables."""
        q_fft, ap_fft, am_fft = self.oper.qapamfft_from_uxuyetafft(
            ux_fft, uy_fft, eta_fft
        )

        return self.bvecfft_from_qapamfft(q_fft, ap_fft, am_fft)

    def compute_qda_energies_fft(self):
        """Compute quadratic geostrophic, divergent and ageostrophic energies."""
        bvec_fft = self.compute()
        q_fft = bvec_fft[0]
        d_fft = 0.5 * (bvec_fft[1] - bvec_fft[2])
        a_fft = 0.5 * (bvec_fft[1] + bvec_fft[2])

        def energy(var_fft):
            return 0.5 * np.abs(var_fft) ** 2

        return energy(q_fft), energy(d_fft), energy(a_fft)

    def compute_qapam_energies_fft(self):
        """Compute quadratic geostrophic, and ageostrophic (+/-) energies."""
        bvec_fft = self.compute()

        def energy(var_fft):
            return 0.5 * np.abs(var_fft) ** 2

        return map(energy, bvec_fft)


class NormalModeDecomposition(NormalModeBase):
    def __init__(self, output):
        super().__init__(output)

    @cached_property
    def qmat(self):
        """Compute Q matrix to transform q, ap, am (fft) -> b0, b+, b- (fft) with
        dimensions of velocity.

        """
        oper = self.oper
        sigma = self.sigma

        f = float(self.params.f)
        c = self.params.c2**0.5
        KX = oper.KX
        KY = oper.KY
        K = oper.K
        K2 = oper.K2
        K_not0 = oper.K_not0
        ck = c * K_not0

        qmat = np.array(
            [
                [
                    -1j * 2.0**0.5 * ck * KY,
                    +1j * f * KY + KX * sigma,
                    +1j * f * KY - KX * sigma,
                ],
                [
                    +1j * 2.0**0.5 * ck * KX,
                    -1j * f * KX + KY * sigma,
                    -1j * f * KX - KY * sigma,
                ],
                [2.0**0.5 * f * K, c * K2, c * K2],
            ]
        ) / (2.0**0.5 * sigma * K_not0)

        if mpi.rank == 0 or oper.is_sequential:
            qmat[:, :, 0, 0] = 0.0

        return qmat

    def normalmodefft_from_keyfft(self, key):
        """Returns the normal mode decomposition for the state_spect key specified."""

        if key == "div_fft":
            key_modes, normal_mode_vec_fft_x = self.normalmodefft_from_keyfft(
                "px_ux_fft"
            )
            key_modes, normal_mode_vec_fft_y = self.normalmodefft_from_keyfft(
                "py_uy_fft"
            )
            normal_mode_vec_fft = normal_mode_vec_fft_x + normal_mode_vec_fft_y
        else:
            key_modes = np.array([["G", "A", "a"]])
            row_index = {
                "ux_fft": 0,
                "uy_fft": 1,
                "eta_fft": 2,
                "px_ux_fft": 0,
                "px_uy_fft": 1,
                "px_eta_fft": 2,
                "py_ux_fft": 0,
                "py_uy_fft": 1,
                "py_eta_fft": 2,
            }

            r = row_index[key]
            normal_mode_vec_fft = np.einsum(
                "i...,i...->i...", self.qmat[r], self.bvec_fft
            )
            if "px" in key:
                for r in range(3):
                    normal_mode_vec_fft[r] = self.oper.pxffft_from_fft(
                        normal_mode_vec_fft[r]
                    )
            elif "py" in key:
                for r in range(3):
                    normal_mode_vec_fft[r] = self.oper.pyffft_from_fft(
                        normal_mode_vec_fft[r]
                    )

            if "eta" in key:
                normal_mode_vec_fft = normal_mode_vec_fft / self.params.c2**0.5

        return key_modes, normal_mode_vec_fft

    def normalmodephys_from_keyphys(self, key):
        ifft2 = self.oper.ifft2
        key_modes, normal_mode_vec_fft = self.normalmodefft_from_keyfft(
            key + "_fft"
        )
        normal_mode_vec_phys = np.array(
            [ifft2(normal_mode_vec_fft[i]) for i in range(3)]
        )

        return key_modes, normal_mode_vec_phys

    def _group_matrix_using_dict(self, key_matrix, value_matrix, grouping):
        value_dict = dict.fromkeys(grouping, 0.0)
        n1, n2 = key_matrix.shape
        for i in range(n1):
            for j in range(n2):
                k1 = key_matrix[i, j]
                k3 = None
                for k2 in grouping.keys():
                    if k1 in grouping[k2]:
                        k3 = k2
                        break

                if k3 is None:
                    raise KeyError(
                        "Not sure which dyad group " + k1 + " belongs to"
                    )

                value_dict[k3] += value_matrix[i, j]

        new_matrix = np.array([value_dict[k] for k in value_dict.keys()])
        new_keys = np.array([list(value_dict)])
        return new_keys, new_matrix

    def dyad_from_keyfft(self, conjugate=False, *keys_state_spect):
        dyad_group = {
            "GG": ["GG"],
            "AG": ["GA", "AG"],
            "aG": ["Ga", "aG"],
            "AA": ["AA", "Aa", "aA", "aa"],
        }
        k1, k2 = keys_state_spect

        normal_modes = dict()
        if k1 != k2:
            for k in keys_state_spect:
                key_modes, normal_modes[k] = self.normalmodefft_from_keyfft(k)
        else:
            key_modes, normal_modes[k1] = self.normalmodefft_from_keyfft(k1)
            normal_modes[k2] = normal_modes[k1]

        key_modes_mat = np.char.add(key_modes.transpose(), key_modes)
        if conjugate:
            Ni = normal_modes[k1].conj()
            Nj = normal_modes[k2]
        else:
            Ni = normal_modes[k1]
            Nj = normal_modes[k2]
        dyad_mat_fft = np.einsum("i...,j...->ij...", Ni, Nj)
        del (normal_modes, Ni, Nj)
        return self._group_matrix_using_dict(
            key_modes_mat, dyad_mat_fft, dyad_group
        )

    def dyad_from_keyphys(self, *keys_state_phys):
        dyad_group = {
            "GG": ["GG"],
            "AG": ["GA", "AG"],
            "aG": ["Ga", "aG"],
            "AA": ["AA", "Aa", "aA", "aa"],
        }
        k1, k2 = keys_state_phys

        normal_modes = dict()
        if k1 != k2:
            for k in keys_state_phys:
                key_modes, normal_modes[k] = self.normalmodephys_from_keyphys(k)
        else:
            key_modes, normal_modes[k1] = self.normalmodephys_from_keyphys(k1)
            normal_modes[k2] = normal_modes[k1]
        key_modes_mat = np.char.add(key_modes.transpose(), key_modes)
        dyad_mat_phys = np.einsum(
            "i...,j...->ij...", normal_modes[k1], normal_modes[k2]
        )
        del normal_modes
        fft2 = self.oper.fft2
        dyad_mat_fft = np.array(
            [[fft2(dyad_mat_phys[i, j]) for j in range(3)] for i in range(3)]
        )

        for i in range(3):
            for j in range(3):
                self.oper.dealiasing(dyad_mat_fft[i, j])

        del dyad_mat_phys
        return self._group_matrix_using_dict(
            key_modes_mat, dyad_mat_fft, dyad_group
        )

    def triad_from_keyfft(self, *keys_state_spect):
        triad_group = {
            "GGG": ["GGG"],
            "AGG": ["AGG", "GAG", "GGA", "aGG", "GaG", "GGa"],
            "GAAs": ["aaG", "aGa", "Gaa", "AAG", "AGA", "GAA"],
            "GAAd": ["aAG", "AaG", "aGA", "AGa", "GaA", "GAa"],
            "AAA": ["AAA", "aaa", "AAa", "AaA", "aAA", "aaA", "aAa", "Aaa"],
        }
        k1, k2, k3 = keys_state_spect

        key_modes_1, normal_modes_1 = self.normalmodefft_from_keyfft(k1)
        key_modes_23, normal_modes_23 = self.dyad_from_keyfft(False, k2, k3)

        key_modes_mat = np.char.add(key_modes_1.transpose(), key_modes_23)
        triad_mat = np.einsum(
            "i...,j...->ij...", normal_modes_1.conj(), normal_modes_23
        )

        return self._group_matrix_using_dict(
            key_modes_mat, triad_mat, triad_group
        )

    def triad_from_keyfftphys(self, key_state_spect, *keys_state_phys):
        triad_group = {
            "GGG": ["GGG"],
            "AGG": ["AGG", "GAG", "GGA", "aGG", "GaG", "GGa"],
            "GAAs": ["aaG", "aGa", "Gaa", "AAG", "AGA", "GAA"],
            "GAAd": ["aAG", "AaG", "aGA", "AGa", "GaA", "GAa"],
            "AAA": ["AAA", "aaa", "AAa", "AaA", "aAA", "aaA", "aAa", "Aaa"],
        }
        k1 = key_state_spect
        k2, k3 = keys_state_phys

        key_modes_1, normal_modes_1 = self.normalmodefft_from_keyfft(k1)
        key_modes_23, normal_modes_23 = self.dyad_from_keyphys(k2, k3)

        key_modes_mat = np.char.add(key_modes_1.transpose(), key_modes_23)
        triad_mat = np.einsum(
            "i...,j...->ij...", normal_modes_1.conj(), normal_modes_23
        )

        return self._group_matrix_using_dict(
            key_modes_mat, triad_mat, triad_group
        )


class NormalModeDecompositionModified(NormalModeDecomposition):
    def compute(self):
        if self.it_bvec_fft_computed != self.sim.time_stepping.it:
            get_var = self.sim.state.get_var
            #
            # FIXME:Does not work
            #
            # if {'ap_fft', 'am_fft'}.issubset(self.sim.state.keys_state_spect):
            #     q_fft = get_var('q_fft')
            #     ap_fft = get_var('ap_fft')
            #     am_fft = get_var('am_fft')
            #     a_fft = ap_fft + am_fft
            #     bvecrot_fft = self.bvecfft_from_qapamfft(q_fft, a_fft, a_fft)
            # else:
            rot_fft = self.sim.state.get_var("rot_fft")
            uxr_fft, uyr_fft = self.oper.vecfft_from_rotfft(rot_fft)
            eta_fft = get_var("eta_fft")
            bvecrot_fft = self.bvecfft_from_uxuyetafft(uxr_fft, uyr_fft, eta_fft)

            self.bvecrot_fft = bvecrot_fft

        return super().compute()

    def normalmodefft_from_keyfft(self, key):
        """Returns the normal mode decomposition for the state_spect key specified."""
        if key.endswith("urx_fft") or key.endswith("ury_fft"):
            row_index = {
                "urx_fft": 0,
                "ury_fft": 1,
                "px_urx_fft": 0,
                "px_ury_fft": 1,
                "py_urx_fft": 0,
                "py_ury_fft": 1,
            }

            key_modes = np.array([["G", "A", "a"]])
            r = row_index[key]
            normal_mode_vec_fft = np.einsum(
                "i...,i...->i...", self.qmat[r], self.bvecrot_fft
            )
            if "px" in key:
                for r in range(3):
                    normal_mode_vec_fft[r] = self.oper.pxffft_from_fft(
                        normal_mode_vec_fft[r]
                    )
            elif "py" in key:
                for r in range(3):
                    normal_mode_vec_fft[r] = self.oper.pyffft_from_fft(
                        normal_mode_vec_fft[r]
                    )

            return key_modes, normal_mode_vec_fft

        else:
            return super().normalmodefft_from_keyfft(key)
