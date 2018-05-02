# -*- coding: utf-8 -*-
""" Anisotropic (:mod:`fluidsim.base.forcing.anisotropic`)
==========================================================

.. autoclass:: TimeCorrelatedRandomPseudoSpectralAnisotropic
   :members:
   :private-members:

"""
from __future__ import division
from __future__ import print_function

from math import degrees

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
        """This static method is used to complete the *params* container.
        """
        super(
            TimeCorrelatedRandomPseudoSpectral, cls
        )._complete_params_with_default(
            params
        )

        params.forcing._set_child("tcrandom_anisotropic", {"angle": "45°"})

    # def __init__(self, sim):
    #     super(TimeCorrelatedRandomPseudoSpectralAnisotropic, self).__init__(sim)

    #     # To plot forcing 1 mode (1k) Vs time
    #     if (
    #         self.params.forcing.tcrandom.time_correlation
    #         == "based_on_forcing_rate"
    #     ):
    #         if self.key_forced == "rot_fft":
    #             time_correlation = self.forcing_rate ** (-1. / 3)
    #         elif self.key_forced == "ap_fft":
    #             raise NotImplementedError

    #     else:
    #         time_correlation = self.params.forcing.tcrandom.time_correlation

    #     self.t_last_change_1k = 0.0
    #     self.period_save_forcing1k = (1 / 4.) * time_correlation
    #     self.forcing_1k = []
    #     self.time_1k = []

    # def compute(self):
    #     F_fft = super(
    #         TimeCorrelatedRandomPseudoSpectralAnisotropic, self
    #     ).compute()
    #     print('Fa_fft', Fa_fft)

    # # Save forcing first mode forced
    # id0, id1 = np.argwhere(self.COND_NO_F == False)[0]
    # forcing_1k = Fa_fft[id0, id1].real
    # tsim = self.sim.time_stepping.t
    # # print(tsim)
    # if tsim - self.t_last_change_1k >= self.period_save_forcing1k:
    #     self.forcing_1k.append(forcing_1k)
    #     self.time_1k.append(self.sim.time_stepping.t)
    #     self.t_last_change_1k = tsim

    def _compute_cond_no_forcing(self):
        """Computes condition no forcing of the anisotropic case.
        """
        angle = self.angle

        self.kxmin_forcing = np.sin(angle) * self.kmin_forcing
        self.kxmax_forcing = np.sin(angle) * self.kmax_forcing

        self.kymin_forcing = np.cos(angle) * self.kmin_forcing
        self.kymax_forcing = np.cos(angle) * self.kmax_forcing

        if (
            self.kxmax_forcing - self.kxmin_forcing < self.oper.deltakx
            or self.kymax_forcing - self.kymin_forcing < self.oper.deltaky
        ):
            raise ValueError("No forcing modes in one direction.")

        COND_NO_F_KX = np.logical_or(
            self.oper_coarse.KX > self.kxmax_forcing,
            self.oper_coarse.KX < self.kxmin_forcing,
        )

        COND_NO_F_KY = np.logical_or(
            self.oper_coarse.KY > self.kymax_forcing,
            self.oper_coarse.KY < self.kymin_forcing,
        )

        COND_NO_F = np.logical_or(COND_NO_F_KX, COND_NO_F_KY)
        COND_NO_F[self.oper_coarse.shapeK_loc[0] // 2] = True
        COND_NO_F[:, self.oper_coarse.shapeK_loc[1] - 1] = True

        return COND_NO_F

    def plot_forcing_1mode_time(self):
        """Plots the forcing coherence in time."""

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

        theta1 = 90. - degrees(self.angle)
        theta2 = 90.

        KX = self.oper_coarse.KX
        KY = self.oper_coarse.KY

        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        title = (
            pforcing.type
            + "; "
            + r"$nk_{{min}} = {} \delta k_x$; ".format(pforcing.nkmin_forcing)
            + r"$nk_{{max}} = {} \delta k_z$; ".format(pforcing.nkmax_forcing)
            + r"$\theta = {:.0f}^\circ$; ".format(degrees(self.angle))
            + r"Forced modes = {}".format(self.nb_forced_modes)
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
                theta2=theta2
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
                linestyle="-."
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
                linestyle="-."
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
