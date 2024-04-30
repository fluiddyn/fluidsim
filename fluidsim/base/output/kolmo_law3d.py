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

""" Conversion from cartesian coordinates system to spherical coordinate system """


class OperKolmoLaw:
    def __init__(self, X, Y, Z, params):
        self.r = np.sqrt(X**2 +Y**2 + Z**2)
        self.rh = np.sqrt(X**2 + Y**2)
        self.rz = Z
#        self.sin_theta_sin_phi = Y / self.r
#        self.cos_phi_sin_theta  = X / self.r
#        self.sin_theta = self.rh / self.r
#        self.cos_theta = Z / self.r
#        self.sin_phi = Y  / self.rh
#        self.cos_phi = X  / self.rh


    def average_azimutal(self, arr):


        avg_arr = None
        if mpi.nb_proc == 1:
            avg_arr = np.mean(arr, axis=(0,1))
        return avg_arr

        local_sum = np.sum(arr, axis=(0,1))
        if mpi.rank == 0:
            global_arr = np.zeros(self.nz)  # define array to sum on all proc

        for rank in range(mpi.nb_proc):
            if mpi.rank == 0:
                nz_loc = self.nzs_local[rank]  # define size of array on each proc
            if rank == 0 and mpi.rank == 0:
                sum = local_sum  # start the sum on rank 0
            else:
                if mpi.rank == 0:  # sum made on rank 0: receive local_array of rank
                    sum = np.empty(nz_loc)
                    mpi.comm.Recv(sum, source=rank, tag=42 * rank)
                elif mpi.rank == rank:  # send the local array to 0
                    mpi.comm.Send(sum_local, dest=0, tag=42 * rank)
            if mpi.rank == 0:  # construct global sum on 0
                iz_start = self.izs_start[rank]
                global_array[iz_start : iz_start + nz_loc] += sum
        if mpi.rank == 0:
            avg_arr = global_array / self.nazim
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


                fig3, ax3 = self.output.figure_axe()
                self.ax3 = ax3
                ax3.set_xlabel("$r/eta$")
                ax3.set_ylabel("$S2(r)$")
                if scale is None:
                    ax3.set_yscale("log")
                    ax3.set_xscale("log")
                    ax3.set_ylim([0.9, 10.0])
                else:
                    ax3.set_yscale(f"{scale}")
                    ax3.set_xscale("log")
                if slope is not None:
                    ax3.plot(posxprime, check_slope, label=f"$r^{2}$")
                ax3.plot(posx, E_k, label="4*E")
                ax3.plot(posx, pos2y, label=f"S2(r)")
                ax3.set_title(f"tmin={tmin:.2g}, tmax={tmax:.2g}")

                ax3.legend()

    def compute(self):
        """compute the values at one time."""
        state = self.sim.state
        state_phys = state.state_phys
        state_spect = state.state_spect
        keys_state_phys=state.keys_state_phys
        fft = self.sim.oper.fft

        letters = "xyz"

        tf_vi = [state_spect.get_var(f"v{letter}_fft") for letter in letters]
        vel = [state_phys.get_var(f"v{letter}") for letter in letters]
        tf_vjvi = np.empty((3, 3), dtype=object)
        tf_K = None

        if "b" in keys_state_phys:
            b= check_strat(state_phys.get_var("b"))
            tf_b = check_strat(state_spect.get_var("b_fft"))
            b2 = b * b
            tf_b2 = fft(b2)
            tf_bv=[None] * 3
            bv = [item * b for item in vel]
            for index in range(len(bv)):
                tf_bv[index] = fft(bv[index])
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

        J_K_xyz = [None] * 3
        J_A_xyz = [None] * 3
        for ind_i in range(3):
            tmp = tf_vi[ind_i] * tf_K.conj()
            if "b" in keys_state_phys:
                mom = 4 * tf_bv[ind_i].conj() * tf_b + 2 * tf_b2.conj() * tf_vi[ind_i]
                mom.real = 0.0
            for ind_j in range(3):
                tmp += tf_vi[ind_j] * tf_vjvi[ind_i, ind_j].conj()
            tmp.real = 0.0
            J_K_xyz[ind_i] = 4 * self.sim.oper.ifft(tmp)
            if "b" in keys_state_phys:
                J_A_xyz[ind_i] = self.sim.oper.ifft(mom)

        result = {}
        keys = ["r", "theta", "phi"]
        for index, key in enumerate(keys):
            result["J_K_" + key] = self.oper_kolmo_law.average_azimutal(
                J_K_xyz[index]
            )

        return result

    def check_diff_methods(self):
        first_method = compute(self)
        second_method = compute_alt(self)
        if not np.allclose(first_method, second_method):
            raise RuntimeError(
                "Both methods are inconsistent: " " ({self.sim.time_stepping.it = })"
            )
