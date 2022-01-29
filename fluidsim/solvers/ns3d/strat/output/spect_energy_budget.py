"""SEB (:mod:`fluidsim.solvers.ns3d.strat.output.spect_energy_budget`)
======================================================================

.. autoclass:: SpectralEnergyBudgetNS3DStrat
   :members:
   :private-members:

"""

import numpy as np

from warnings import warn

from fluiddyn.util import mpi

from fluidsim.solvers.ns3d.output.spect_energy_budget import (
    SpectralEnergyBudgetNS3D,
)


class SpectralEnergyBudgetNS3DStrat(SpectralEnergyBudgetNS3D):
    r"""Spectral energy budget of ns3d.strat.

    .. |vv| mathmacro:: \textbf{v}

    .. |kk| mathmacro:: \textbf{k}

    .. |convK2A| mathmacro:: C_{K\rightarrow A}

    .. |bnabla| mathmacro:: \boldsymbol{\nabla}

    .. |bomega| mathmacro:: \boldsymbol{\omega}

    Notes
    -----

    .. math::

      d_t E_K(\kk) = T_K(\kk) - \convK2A(\kk) - D_K(\kk),

      d_t E_A(\kk) = T_A(\kk) + \convK2A(\kk) - D_A(\kk),

    where :math:`E_K(\kk) = |\hat{\vv}|^2/2` and :math:`E_A(\kk) = |\hat{b}|^2/(2N^2)`.

    The transfer terms are

    .. math::

      T_K(\kk) = \Re \left(\hat{\vv}^* \cdot P_\perp\widehat{\vv \times \bomega} \right),

      T_A(\kk) = - \Re \left(\hat{b}^* \widehat{\vv \cdot \bnabla b} \right) / N^2.

    By definition, we have to have :math:`\sum T_K(\kk) = \sum T_A(\kk) = 0`.

    The conversion term is equal to :math:`\convK2A(\kk) = -\Re(\hat{v_z}^*
    \hat{b})` and the dissipative terms are

    .. math::

      D_K(\kk) = f_{dK}(\kk) E_K(\kk),

      D_A(\kk) = f_{dA}(\kk) E_A(\kk),

    where :math:`f_{dK}(\kk)` and :math:`f_{dA}(\kk)` are the dissipation
    frequency depending of the wavenumber and the viscous coefficients.

    """

    def compute(self):

        results = super().compute()

        state = self.sim.state
        N = self.sim.params.N

        oper = self.sim.oper

        state_spect = state.state_spect

        vz_fft = state_spect.get_var("vz_fft")

        state_phys = state.state_phys

        f_d, f_d_hypo = self.sim.compute_freq_diss()
        del f_d_hypo

        b_fft = state_spect.get_var("b_fft")

        results.update(
            self.compute_spectra("diss_A", 1 / N**2 * f_d * abs(b_fft) ** 2)
        )

        results.update(
            self.compute_spectra("conv_K2A", -np.real(vz_fft.conj() * b_fft))
        )
        del vz_fft

        vx = state_phys.get_var("vx")
        vy = state_phys.get_var("vy")
        vz = state_phys.get_var("vz")

        fb_fft = (
            -1
            / N**2
            * oper.div_vb_fft_from_vb(vx, vy, vz, state_phys.get_var("b"))
        )
        del vx, vy, vz

        results.update(
            self.compute_spectra("transfer_A", np.real(b_fft.conj() * fb_fft))
        )

        transfer_A = results["transfer_A"]
        if mpi.rank == 0 and transfer_A.sum() > 1e-12:
            warn(
                f"spect_energy_budg: transfer_A.sum() is too big {transfer_A.sum()}"
            )

        return results

    def plot_fluxes(
        self, tmin=None, tmax=None, key_k="kh", ax=None, plot_conversion=True
    ):

        data = self.compute_fluxes_mean(tmin, tmax)

        k_plot = data[key_k]
        deltak = k_plot[1]
        k_plot += deltak / 2

        key_flux = key_k[1] + "flux_"
        flux_Kh = data[key_flux + "transfer_Kh"]
        flux_Kz = data[key_flux + "transfer_Kz"]
        flux_A = data[key_flux + "transfer_A"]
        DKh = data[key_flux + "diss_Kh"]
        DKz = data[key_flux + "diss_Kz"]
        DA = data[key_flux + "diss_A"]
        if plot_conversion:
            CK2A = data[key_flux + "conv_K2A"]

        flux_K = flux_Kh + flux_Kz
        flux_tot = flux_K + flux_A
        DK = DKh + DKz
        D = DK + DA

        eps = D[-1]

        if ax is None:
            fig, ax = self.output.figure_axe()

        xlabel = "k_" + key_k[1]
        ylabel = rf"$\Pi({xlabel})/\epsilon$"
        ax.set_xlabel(f"${xlabel}$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"spectral fluxes\n{self.output.summary_simul}")

        def _plot(x, y, code, label, linewidth=None):
            ax.semilogx(x, y / eps, code, label=label, linewidth=linewidth)

        _plot(k_plot, flux_tot, "k", r"$\Pi/\epsilon$", linewidth=2)
        _plot(k_plot, D, "k--", r"$D/\epsilon$", linewidth=2)
        _plot(k_plot, flux_tot + D, "k:", r"$(\Pi+D)/\epsilon$")
        _plot(k_plot, flux_K, "r", r"$\Pi_K/\epsilon$")
        _plot(k_plot, flux_A, "b", r"$\Pi_A/\epsilon$")
        _plot(k_plot, DK, "r--", r"$D_K/\epsilon$")
        _plot(k_plot, DA, "b--", r"$D_A/\epsilon$")
        if plot_conversion:
            _plot(k_plot, CK2A, "m", r"$B/\epsilon$", linewidth=2)

        ax.legend()
