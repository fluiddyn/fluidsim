# -*- coding: utf-8 -*-
""" Anisotropic (:mod:`fluidsim.base.forcing.anisotropic`)
==========================================================

.. autoclass:: TimeCorrelatedRandomPseudoSpectralAnisotropic
   :members:
   :private-members:

"""

from math import degrees
from math import pi

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fluidsim.base.forcing.specific import TimeCorrelatedRandomPseudoSpectral


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
            "tcrandom_anisotropic", {"angle": "45°", "kz_negative_enable": False}
        )

    def __init__(self, sim):
        super().__init__(sim)

        if self.params.forcing.normalized.type == "particular_k":
            raise NotImplementedError

    def _compute_cond_no_forcing(self):
        """Computes condition no forcing of the anisotropic case."""
        angle = self.angle

        self.kxmin_forcing = np.sin(angle) * self.kmin_forcing
        self.kxmax_forcing = np.sin(angle) * self.kmax_forcing

        self.kymin_forcing = np.cos(angle) * self.kmin_forcing
        self.kymax_forcing = np.cos(angle) * self.kmax_forcing

        COND_NO_F_KX = np.logical_or(
            self.oper_coarse.KX > self.kxmax_forcing,
            self.oper_coarse.KX < self.kxmin_forcing,
        )

        COND_NO_F_KY = np.logical_or(
            self.oper_coarse.KY > self.kymax_forcing,
            self.oper_coarse.KY < self.kymin_forcing,
        )

        if self.params.forcing.tcrandom_anisotropic.kz_negative_enable:
            COND_NO_F_KY = np.logical_and(
                COND_NO_F_KY,
                np.logical_or(
                    self.oper_coarse.KY < -self.kymax_forcing,
                    self.oper_coarse.KY > -self.kymin_forcing,
                ),
            )

        COND_NO_F = np.logical_or(COND_NO_F_KX, COND_NO_F_KY)
        COND_NO_F[self.oper_coarse.shapeK_loc[0] // 2] = True
        COND_NO_F[:, self.oper_coarse.shapeK_loc[1] - 1] = True
        return COND_NO_F

    # def plot_forcing_1mode_time(self):
    #     """Plots the forcing coherence in time."""

    def plot_forcing_region(self):
        """Plots the forcing region"""
        pforcing = self.params.forcing
        # oper = self.oper

        kxmin_forcing = self.kxmin_forcing
        kxmax_forcing = self.kxmax_forcing
        kymin_forcing = self.kymin_forcing
        kymax_forcing = self.kymax_forcing

        # Define forcing region
        coord_x = kxmin_forcing
        coord_y = kymin_forcing
        width = kxmax_forcing - kxmin_forcing
        height = kymax_forcing - kymin_forcing

        theta1 = 90.0 - degrees(self.angle)
        theta2 = 90.0

        KX = self.oper_coarse.KX
        KY = self.oper_coarse.KY

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
        ax.set_xlim([abs(KX).min(), abs(KX).max()])
        ax.set_ylim([abs(KY).min(), abs(KY).max()])

        # Set ticks 10% of the KX.max and KY.max
        factor = 0.1
        sep_x = abs(KX).max() * factor
        sep_y = abs(KY).max() * factor
        nb_deltakx = int(sep_x // self.oper.deltakx)
        nb_deltaky = int(sep_y // self.oper.deltaky)

        if not nb_deltakx:
            nb_deltakx = 1
        if not nb_deltaky:
            nb_deltaky = 1

        xticks = np.arange(
            abs(KX).min(), abs(KX).max(), nb_deltakx * self.oper.deltakx
        )
        yticks = np.arange(
            abs(KY).min(), abs(KY).max(), nb_deltaky * self.oper.deltaky
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
                width=abs(KX).max() * 0.5,
                height=abs(KX).max() * 0.5,
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
        ax.plot([0, kxmin_forcing], [0, kymin_forcing], color="k", linewidth=1)
        ax.plot(
            [kxmin_forcing, kxmin_forcing],
            [0, kymin_forcing],
            "k--",
            linewidth=0.8,
        )
        ax.plot(
            [kxmax_forcing, kxmax_forcing],
            [0, kymin_forcing],
            "k--",
            linewidth=0.8,
        )
        ax.plot(
            [0, kxmin_forcing],
            [kymin_forcing, kymin_forcing],
            "k--",
            linewidth=0.8,
        )
        ax.plot(
            [0, kxmin_forcing],
            [kymax_forcing, kymax_forcing],
            "k--",
            linewidth=0.8,
        )

        # Plot forced modes in red
        indices_forcing = np.argwhere(self.COND_NO_F == False)
        for i, index in enumerate(indices_forcing):
            ax.plot(
                KX[0, index[1]],
                KY[index[0], 0],
                "ro",
                label="Forced mode" if i == 0 else "",
            )

        # Location labels 0.8% the length of the axis
        factor = 0.008
        loc_label_y = abs(KY).max() * factor
        loc_label_x = abs(KX).max() * factor

        ax.text(loc_label_x + kxmin_forcing, loc_label_y, r"$k_{x,min}$")
        ax.text(loc_label_x + kxmax_forcing, loc_label_y, r"$k_{x,max}$")
        ax.text(loc_label_x, kymin_forcing + loc_label_y, r"$k_{z,min}$")
        ax.text(loc_label_x, kymax_forcing + loc_label_y, r"$k_{z,max}$")

        # Location label angle \theta
        factor_x = 0.015
        factor_y = 0.15
        loc_label_y = abs(KY).max() * factor_y
        loc_label_x = abs(KX).max() * factor_x

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
        params.forcing.available_types.append(cls.tag)
        params.forcing._set_child(
            cls.tag,
            {
                "angle": "45°",
                "delta_angle": "10°",
                "thetaf_min": None,
                "thetaf_max": None,
                "kz_negative_enable": True,
                "kf_min": 1.0,
                "kf_max": 2.0,
            },
        )

    def __init__(self, sim):
        super().__init__(sim)

        self.params.forcing.tcrandom_anisotropic.thetaf_min = (
            self.params.forcing.tcrandom_anisotropic.angle
            - 0.5 * self.params.forcing.tcrandom_anisotropic.delta_angle
        )
        self.params.forcing.tcrandom_anisotropic.thetaf_max = (
            self.params.forcing.tcrandom_anisotropic.angle
            + 0.5 * self.params.forcing.tcrandom_anisotropic.delta_angle
        )

    def _compute_cond_no_forcing(self):
        """Computes condition no forcing of the tcrandom_anisotropic case."""
        angle = self.params.forcing.tcrandom_anisotropic.angle
        delta_angle = self.params.forcing.tcrandom_anisotropic.delta_angle
        kf_min = self.params.forcing.tcrandom_anisotropic.kf_min
        kf_max = self.params.forcing.tcrandom_anisotropic.kf_max

        K = np.sqrt(
            self.oper_coarse.Kx ** 2
            + self.oper_coarse.Ky ** 2
            + self.oper_coarse.Kz ** 2
        )
        K_nozero = K.copy()
        K_nozero[K_nozero == 0] = 1e-14

        theta = np.arccos(self.oper_coarse.Kz / K_nozero)
        del K_nozero

        COND_NO_F_K = np.logical_or(K > kf_max, K < kf_min)

        COND_NO_F_THETA = np.logical_or(
            theta > angle + 0.5 * delta_angle, theta < angle - 0.5 * delta_angle
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
