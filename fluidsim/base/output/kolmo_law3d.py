"""Kolmogorov law 3d (:mod:`fluidsim.base.output.kolmo_law3d`)
==============================================================

Provides:

.. autoclass:: KolmoLaw
   :members:
   :private-members:

"""
import itertools

import numpy as np

from fluiddyn.util import mpi

from .base import SpecificOutput


class OperKolmoLaw:
    def __init__(self, X, Y, Z, params):
        self.cos_theta_cos_phi = ...
        self.cos_theta_sin_phi = ...
        self.sin_theta = ...
        self.sin_phi = ...
        self.cos_phi = ...

        self.rh = ...
        self.rz = ...

    def vec_rthetaphi_from_vec_xyz(self, vx, vy, vz):

        v_r = ...

        v_theta = (
            self.cos_theta_cos_phi * vx
            + self.cos_theta_sin_phi * vy
            - self.sin_theta * vz
        )

        v_phi = self.sin_phi * vx + self.cos_phi * vy

        return v_r, v_theta, v_phi

    def average_azimutal(arr):
        self.rh
        self.rz
        avg_arr = None
        mpi
        return avg_arr


class KolmoLaw(SpecificOutput):
    r"""Kolmogorov law 3d.

    .. |J| mathmacro:: {\mathbf J}
    .. |v| mathmacro:: {\mathbf v}
    .. |x| mathmacro:: {\mathbf x}
    .. |r| mathmacro:: {\mathbf r}
    .. |Sum| mathmacro:: \sum_{\mathbf k}
    .. |bnabla| mathmacro:: \boldsymbol{\nabla}
    .. |Int| mathmacro:: \displaystyle\int
    .. |epsK| mathmacro:: \varepsilon_K
    .. |epsA| mathmacro:: \varepsilon_A
    .. |dd| mathmacro:: \mathrm{d}

    We want to test the prediction :

    .. math::

        \bnabla \cdot \left( \J_K + \J_A \right) = -4 \left( \epsK + \epsA \right),

    where

    .. math::

        \J_K(\r) \equiv
          \left\langle | \delta \v |^2 \delta \v \right\rangle_\x, \\
        \J_A(\r) \equiv
          \frac{1}{N^2} \left\langle | \delta b |^2 \delta \v \right\rangle_\x.

    This output saves the components in the spherical basis of the vectors
    :math:`\J_\alpha` averaged over the azimutal angle (i.e. as a function of
    :math:`r_h` and :math:`r_z`).

    We can take the example of the quantity :math:`\langle | \delta b |^2
    \delta \v \rangle_\x` to explain how these quantities are computed. Using
    the relation

    .. math::

        \left\langle a' b \right\rangle_\x(\r)
        =\left\langle a(\x+\r)b(\x) \right\rangle_x(\r)\\
        =\Int (a(\x-(-\r))b(x))\dd \x \\
        =a*b(-r) \\
        =TF^{-1} \left\{ \hat{a} \hat{b}^* \right\}(\r),

    it is easy to show that

    .. math::

        \left\langle |\delta b|^2 \delta \v \right\rangle_\x(\r)  = \left\langle (b'^2-bb'+b^2)\v' \right\rangle_\x(\r)-\left\langle (b'^2-bb'+b^2)\v \right\rangle_\x(\r)

        =TF^{-1} \left\{ \widehat{b^2} \hat{\v}^* \right\}(\r) - TF^{-1} \left\{ 2\hat{b} \widehat{b\v}^* \right\}(\r) + \left\langle b^2 \v\right\rangle_\x(\r)
        -\left\langle b'^2 \v \right\rangle_\x(\r) + TF^{-1} \left\{ 2\widehat{b^*} \widehat{b\v}\right\}(\r) - TF^{-1} \left\{ \widehat{b^2}^* \hat{\v} \right\}(\r)
        \\ \\
        \left\langle b^2 \v\right\rangle_\x(\r)= \left\langle b'^2 \v\right\rangle_\x(\r) \text{ with isotropy and } (ab^*)^*=a^*b
        \\ \\
        = TF^{-1} \left\{ \left(\widehat{b^2}^* \hat{\v}\right)^* \right\}(\r) - TF^{-1} \left\{ 2\hat{b} \widehat{b\v}^* \right\}(\r) + TF^{-1} \left\{ \left(2\hat{b} \widehat{b\v}^*\right)^* \right\}(\r)
        -  TF^{-1} \left\{ \widehat{b^2}^* \hat{\v}^* \right\}(\r)
        \\ \\
        ( z-z*=2i \Im(z) )
        \\ \\
        =TF^{-1} \left\{ i\Im \left[ 4 \widehat{(b \v)}^* \hat{b} + 2 \widehat{(b^2)}^* \hat{\v} \right] \right\}

    """

    _tag = "kolmo_law"
    _name_file = "kolmo_law.h5"

    @classmethod
    def _complete_params_with_default(cls, params):
        params.output.periods_save._set_attrib(cls._tag, 0)

    def __init__(self, output):
        params = output.sim.params

        # dict containing rh and rz
        # TODO: complete arrays_1st_time
        try:
            period_save_kolmo_law = params.output.periods_save.kolmo_law
        except AttributeError:
            period_save_kolmo_law = 0.0
        if period_save_kolmo_law != 0.0:
            X, Y, Z = output.sim.oper.get_XYZ_loc()
            self.oper_kolmo_law = OperKolmoLaw(X, Y, Z, params)
            arrays_1st_time = {
                "rh": self.oper_kolmo_law.rh,
                "rz": self.oper_kolmo_law.rz,
            }
        else:
            arrays_1st_time = None

        super().__init__(
            output,
            period_save=period_save_kolmo_law,
            arrays_1st_time=arrays_1st_time,
        )

    def compute(self):
        """compute the values at one time."""
        state = self.sim.state
        state_phys = state.state_phys
        state_spect = state.state_spect

        fft = self.sim.oper.fft

        letters = "xyz"

        tf_vi = [state_spect.get_var(f"v{letter}_fft") for letter in letters]
        tf_vjvi = np.empty((3, 3), dtype=object)
        tf_K = None

        for index, letter in enumerate(letters):
            vi = state_phys.get_var("v" + letter)
            vi2 = vi * vi
            tf_vjvi[index, index] = tmp = fft(vi2)
            if tf_K is None:
                tf_K = tmp
            else:
                tf_K += tmp

        for ind_i, ind_j in itertools.combinations(range(3), 2):
            letter_i = letters[ind_i]
            letter_j = letters[ind_j]
            vi = state_phys.get_var("v" + letter_i)
            vj = state_phys.get_var("v" + letter_j)
            tf_vjvi[ind_i, ind_j] = tf_vjvi[ind_j, ind_i] = fft(vi * vj)

        J_xyz = [None] * 3

        for ind_i in range(3):
            tmp = tf_vi[ind_i] * tf_K.conj()
            for ind_j in range(3):
                tmp += tf_vi[ind_j] * tf_vjvi[ind_i, ind_j].conj()
            tmp.real = 0.0
            J_xyz[ind_i] = 4 * self.sim.oper.ifft(tmp)

        J_rthetaphi = self.oper_kolmo_law.vec_rthetaphi_from_vec_xyz(*J_xyz)

        result = {}
        keys = ["r", "theta", "phi"]
        for index, key in enumerate(keys):
            result["J_K_" + key] = self.oper_kolmo_law.average_azimutal(
                J_rthetaphi[index]
            )

        return result
