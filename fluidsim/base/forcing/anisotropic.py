# -*- coding: utf-8 -*-
""" Anisotropic (:mod:`fluidsim.base.forcing.anisotropic`)
==========================================================

.. autoclass:: TimeCorrelatedRandomPseudoSpectralAnisotropic
   :members:
   :private-members:

"""

from math import degrees
from math import pi, radians

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fluiddyn.calcul.easypyfft import fftw_grid_size

from fluidsim.base.forcing.specific import TimeCorrelatedRandomPseudoSpectral


def ensure_radians(angle):
    if isinstance(angle, str):
        if angle.endswith("°"):
            angle = radians(float(angle[:-1]))
        else:
            raise ValueError(
                "Angle should be a string with \n"
                + "the degree symbol or a float in radians"
            )
    return angle


class TimeCorrelatedRandomPseudoSpectralAnisotropic(
    TimeCorrelatedRandomPseudoSpectral
):
    """Random normalized anisotropic forcing.

    .. inheritance-diagram:: TimeCorrelatedRandomPseudoSpectralAnisotropic

    """

    tag = "tcrandom_anisotropic"

    @classmethod
    def _complete_params_with_default(cls, params):
        """This static method is used to complete the *params* container."""
        super(
            TimeCorrelatedRandomPseudoSpectral, cls
        )._complete_params_with_default(params)

        params.forcing._set_child(
            "tcrandom_anisotropic",
            {"angle": "45°", "delta_angle": None, "kz_negative_enable": False},
        )

    def __init__(self, sim):
        super().__init__(sim)

        if self.params.forcing.normalized.type == "particular_k":
            raise NotImplementedError

    def _create_params_coarse(self, fft_size):

        params_coarse = super()._create_params_coarse(fft_size)

        self.angle = angle = ensure_radians(self.params.forcing[self.tag].angle)

        tmp = self.params.forcing.tcrandom_anisotropic
        try:
            delta_angle = tmp.delta_angle
        except AttributeError:
            # loading old simul with delta_angle
            delta_angle = None
        else:
            delta_angle = ensure_radians(delta_angle)

        if delta_angle is None:
            self.khmax_forcing = np.sin(angle) * self.kmax_forcing
            self.kvmax_forcing = np.cos(angle) * self.kmax_forcing
        else:
            self.khmax_forcing = (
                np.sin(angle + 0.5 * delta_angle) * self.kmax_forcing
            )
            self.kvmax_forcing = (
                np.cos(angle - 0.5 * delta_angle) * self.kmax_forcing
            )

        if hasattr(params_coarse.oper, "nz"):
            # 3d
            kymax_forcing = self.khmax_forcing
        else:
            # 2d
            kymax_forcing = self.kvmax_forcing

        # The "+ 1" aims to give some gap between the kxmax and
        # the boundary of the oper_coarse.
        try:
            params_coarse.oper.nx = 2 * fftw_grid_size(
                int(self.khmax_forcing / self.oper.deltakx) + 1
            )
        except AttributeError:
            pass

        try:
            params_coarse.oper.ny
        except AttributeError:
            pass
        else:
            params_coarse.oper.ny = 2 * fftw_grid_size(
                int(kymax_forcing / self.oper.deltaky) + 1
            )

        try:
            params_coarse.oper.nz
        except AttributeError:
            pass
        else:
            params_coarse.oper.nz = 2 * fftw_grid_size(
                int(self.kvmax_forcing / self.oper.deltakz) + 1
            )

        return params_coarse

    def _compute_cond_no_forcing(self):
        """Computes condition no forcing of the anisotropic case."""
        angle = self.angle

        tmp = self.params.forcing.tcrandom_anisotropic
        try:
            delta_angle = tmp.delta_angle
        except AttributeError:
            # loading old simul with delta_angle
            delta_angle = None

        kf_min = self.kmin_forcing
        kf_max = self.kmax_forcing

        try:
            self.params.oper.nz
        except AttributeError:
            ndim = 2
        else:
            ndim = 3

        if delta_angle is None:

            self.khmin_forcing = np.sin(angle) * self.kmin_forcing
            self.kvmin_forcing = np.cos(angle) * self.kmin_forcing

            if ndim == 2:
                Kh = self.oper_coarse.KX
                Kv = self.oper_coarse.KY
            else:
                Kh = np.sqrt(self.oper_coarse.Kx ** 2 + self.oper_coarse.Ky ** 2)
                Kv = self.oper_coarse.Kz

            COND_NO_F_KH = np.logical_or(
                Kh > self.khmax_forcing,
                Kh < self.khmin_forcing,
            )

            COND_NO_F_KV = np.logical_or(
                self.oper_coarse.KY > self.kvmax_forcing,
                self.oper_coarse.KY < self.kvmin_forcing,
            )

            if self.params.forcing.tcrandom_anisotropic.kz_negative_enable:
                COND_NO_F_KV = np.logical_and(
                    COND_NO_F_KV,
                    np.logical_or(
                        Kv < -self.kvmax_forcing,
                        Kv > -self.kvmin_forcing,
                    ),
                )

            COND_NO_F = np.logical_or(COND_NO_F_KH, COND_NO_F_KV)
            COND_NO_F[self.oper_coarse.shapeK_loc[0] // 2] = True
            COND_NO_F[:, self.oper_coarse.shapeK_loc[1] - 1] = True

        else:

            if ndim == 2:
                K = np.sqrt(self.oper_coarse.KX ** 2 + self.oper_coarse.KY ** 2)
                Kv = self.oper_coarse.KY
            else:

                K = np.sqrt(
                    self.oper_coarse.Kx ** 2
                    + self.oper_coarse.Ky ** 2
                    + self.oper_coarse.Kz ** 2
                )
                Kv = self.oper_coarse.Kz

            K_nozero = K.copy()
            K_nozero[K_nozero == 0] = 1e-14

            theta = np.arccos(Kv / K_nozero)
            del K_nozero

            COND_NO_F_K = np.logical_or(K > kf_max, K < kf_min)

            COND_NO_F_THETA = np.logical_or(
                theta > angle + 0.5 * delta_angle,
                theta < angle - 0.5 * delta_angle,
            )

            if self.params.forcing.tcrandom_anisotropic.kz_negative_enable:
                COND_NO_F_THETA = np.logical_and(
                    COND_NO_F_THETA,
                    np.logical_or(
                        theta < pi - angle - 0.5 * delta_angle,
                        theta > pi - angle + 0.5 * delta_angle,
                    ),
                )

            COND_NO_F = np.logical_or(COND_NO_F_K, COND_NO_F_THETA)
            COND_NO_F[self.oper_coarse.shapeK_loc[0] // 2] = True
            COND_NO_F[:, self.oper_coarse.shapeK_loc[1] - 1] = True

        return COND_NO_F

    def plot_forcing_region(self):
        """Plots the forcing region"""
        pforcing = self.params.forcing

        khmin_forcing = self.khmin_forcing
        khmax_forcing = self.khmax_forcing
        kvmin_forcing = self.kvmin_forcing
        kvmax_forcing = self.kvmax_forcing

        tmp = self.params.forcing.tcrandom_anisotropic
        try:
            delta_angle = tmp.delta_angle
        except AttributeError:
            # loading old simul with delta_angle
            delta_angle = None

        if delta_angle is not None:
            # TODO: implement (also for 3d solvers) + test in 3d
            raise NotImplementedError

        try:
            self.params.oper.nz
        except AttributeError:
            ndim = 2
        else:
            ndim = 3

        # Define forcing region
        coord_x = khmin_forcing
        coord_y = kvmin_forcing
        width = khmax_forcing - khmin_forcing
        height = kvmax_forcing - kvmin_forcing

        theta1 = 90.0 - degrees(self.angle)
        theta2 = 90.0

        if ndim == 2:
            Kh = self.oper_coarse.KX
            Kv = self.oper_coarse.KY
        else:
            Kh = np.sqrt(self.oper_coarse.Kx ** 2 + self.oper_coarse.Ky ** 2)
            Kv = self.oper_coarse.Kz

        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        title = (
            pforcing.type
            + "; "
            + fr"$nk_{{min}} = {pforcing.nkmin_forcing} \delta k_x$; "
            + fr"$nk_{{max}} = {pforcing.nkmax_forcing} \delta k_z$; "
            + r"$\theta = {:.0f}^\circ$; ".format(degrees(self.angle))
            + fr"Forced modes = {self.nb_forced_modes}"
        )

        ax.set_title(title)
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_z$")

        # Parameters figure
        ax.set_xlim([abs(Kh).min(), abs(Kh).max()])
        ax.set_ylim([abs(Kv).min(), abs(Kv).max()])

        # Set ticks 10% of the KX.max and KY.max
        factor = 0.1
        sep_x = abs(Kh).max() * factor
        sep_y = abs(Kv).max() * factor
        nb_deltakx = int(sep_x // self.oper.deltakx)
        nb_deltaky = int(sep_y // self.oper.deltaky)

        if not nb_deltakx:
            nb_deltakx = 1
        if not nb_deltaky:
            nb_deltaky = 1

        xticks = np.arange(
            abs(Kh).min(), abs(Kh).max(), nb_deltakx * self.oper.deltakx
        )
        yticks = np.arange(
            abs(Kv).min(), abs(Kv).max(), nb_deltaky * self.oper.deltaky
        )
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.add_patch(
            patches.Rectangle(
                xy=(coord_x, coord_y), width=width, height=height, fill=False
            )
        )

        # width and height arc 50% the length of the axis
        ax.add_patch(
            patches.Arc(
                xy=(0, 0),
                width=abs(Kh).max() * 0.5,
                height=abs(Kh).max() * 0.5,
                angle=0,
                theta1=theta1,
                theta2=theta2,
            )
        )

        # Plot arc kmin and kmax
        ax.add_patch(
            patches.Arc(
                xy=(0, 0),
                width=2 * self.kmin_forcing,
                height=2 * self.kmin_forcing,
                angle=0,
                theta1=0,
                theta2=90.0,
                linestyle="-.",
            )
        )
        ax.add_patch(
            patches.Arc(
                xy=(0, 0),
                width=2 * self.kmax_forcing,
                height=2 * self.kmax_forcing,
                angle=0,
                theta1=0,
                theta2=90.0,
                linestyle="-.",
            )
        )

        # Plot lines angle & lines forcing region
        ax.plot([0, khmin_forcing], [0, kvmin_forcing], color="k", linewidth=1)
        ax.plot(
            [khmin_forcing, khmin_forcing],
            [0, kvmin_forcing],
            "k--",
            linewidth=0.8,
        )
        ax.plot(
            [khmax_forcing, khmax_forcing],
            [0, kvmin_forcing],
            "k--",
            linewidth=0.8,
        )
        ax.plot(
            [0, khmin_forcing],
            [kvmin_forcing, kvmin_forcing],
            "k--",
            linewidth=0.8,
        )
        ax.plot(
            [0, khmin_forcing],
            [kvmax_forcing, kvmax_forcing],
            "k--",
            linewidth=0.8,
        )

        # Plot forced modes in red
        indices_forcing = np.argwhere(self.COND_NO_F == False)
        for i, index in enumerate(indices_forcing):
            ax.plot(
                Kh[0, index[1]],
                Kv[index[0], 0],
                "ro",
                label="Forced mode" if i == 0 else "",
            )

        # Location labels 0.8% the length of the axis
        factor = 0.008
        loc_label_y = abs(Kv).max() * factor
        loc_label_x = abs(Kh).max() * factor

        ax.text(loc_label_x + khmin_forcing, loc_label_y, r"$k_{x,min}$")
        ax.text(loc_label_x + khmax_forcing, loc_label_y, r"$k_{x,max}$")
        ax.text(loc_label_x, kvmin_forcing + loc_label_y, r"$k_{z,min}$")
        ax.text(loc_label_x, kvmax_forcing + loc_label_y, r"$k_{z,max}$")

        # Location label angle \theta
        factor_x = 0.015
        factor_y = 0.15
        loc_label_y = abs(Kv).max() * factor_y
        loc_label_x = abs(Kh).max() * factor_x

        ax.text(loc_label_x, loc_label_y, r"$\theta$")

        ax.grid(linestyle="--", alpha=0.4)
        ax.legend()


class TimeCorrelatedRandomPseudoSpectralAnisotropic3D(
    TimeCorrelatedRandomPseudoSpectralAnisotropic
):
    """Random normalized anisotropic forcing.

    .. inheritance-diagram:: TimeCorrelatedRandomPseudoSpectralAnisotropic3D

    """

    tag = "tcrandom_anisotropic"

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)
        params.forcing.tcrandom_anisotropic.delta_angle = "10°"
