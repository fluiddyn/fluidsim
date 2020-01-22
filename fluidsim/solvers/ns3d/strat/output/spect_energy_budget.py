"""SEB (:mod:`fluidsim.solvers.ns3d.strat.output.spect_energy_budget`)
======================================================================

.. autoclass:: SpectralEnergyBudgetNS3DStrat
   :members:
   :private-members:

"""

import numpy as np
import h5py

from fluiddyn.util import mpi
from fluidfft.fft3d.operators import vector_product

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
        compute_spectrum_kzkh = oper.compute_spectrum_kzkh

        state_spect = state.state_spect

        vx_fft = state_spect.get_var("vx_fft")
        vy_fft = state_spect.get_var("vy_fft")
        vz_fft = state_spect.get_var("vz_fft")

        state_phys = state.state_phys
        vx = state_phys.get_var("vx")
        vy = state_phys.get_var("vy")
        vz = state_phys.get_var("vz")

        fft3d = self.oper.fft3d

        f_d, f_d_hypo = self.sim.compute_freq_diss()

        # todo: f_d_hypo

        del vx_fft, vy_fft

        b_fft = state_spect.get_var("b_fft")

        results.update(
            self.compute_spectra("diss_A", 1 / N ** 2 * f_d * abs(b_fft) ** 2)
        )

        results.update(
            self.compute_spectra("conv_K2A", -np.real(vz_fft.conj() * b_fft))
        )
        del vz_fft

        fb_fft = (
            -1
            / N ** 2
            * oper.div_vb_fft_from_vb(vx, vy, vz, state_phys.get_var("b"))
        )
        del vx, vy, vz

        results.update(
            self.compute_spectra("transfer_A", np.real(b_fft.conj() * fb_fft))
        )

        for key, value in results.items():
            if key.startswith("transfer_A"):
                assert value.sum() < 1e-14

        return results

    def compute_fluxes_mean(self, tmin=None, tmax=None):

        with h5py.File(self.path_file, "r") as file:
            keys_saved = [
                key
                for key in file.keys()
                if key not in ("times", "info_simul")
                and not key.startswith("k")
                and not any(key.endswith("_k" + letter) for letter in "xyz")
            ]

        data = {}
        for key in keys_saved:
            data.update(self.load_mean(tmin, tmax, key))

        kz = data["kz"]
        kh = data["kh"]

        dict_results = {"kz": kz, "kh": kh}

        deltakz = kz[1]
        deltakh = kh[1]

        for key in keys_saved:
            spectrum = data[key]
            flux = deltakz * spectrum.sum(0)
            flux = deltakh * np.cumsum(flux)
            if key.startswith("transfer"):
                flux *= -1

            key_flux = "hflux_" + key
            dict_results.update({key_flux: flux})

            flux = deltakh * spectrum.sum(1)
            flux = deltakz * np.cumsum(flux)
            if key.startswith("transfer"):
                flux *= -1

            key_flux = "zflux_" + key
            dict_results.update({key_flux: flux})

        return dict_results

    def plot_fluxes(self, tmin=None, tmax=None, key_k="kh", ax=None):

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
        CK2A = data[key_flux + "conv_K2A"]

        # errors in ns3d/spect_energy_budget/compute?
        # N = self.params.N
        # flux_A = -flux_A / N ** 2
        # CK2A = -CK2A

        flux_K = flux_Kh + flux_Kz
        flux = flux_K + flux_A
        DK = DKh + DKz
        D = DK + DA

        eps = D[-1]

        if ax is None:
            fig, ax = self.output.figure_axe()

        xlbl = "k_" + key_k[1]
        ylbl = "$\Pi(" + xlbl + ")/\epsilon$"
        xlbl = "$" + xlbl + "$"
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(
            f"spectral fluxes, solver {self.output.name_solver}, nx = {self.nx:5d}"
        )

        ax.semilogx(k_plot, flux / eps, "k", linewidth=2, label="$\Pi/\epsilon$")
        ax.semilogx(k_plot, D / eps, "k--", linewidth=2, label="$D/\epsilon$")
        ax.semilogx(k_plot, DA / eps, "k-.", linewidth=2, label="$D_A/\epsilon$")
        ax.semilogx(k_plot, (flux + D) / eps, "k:", label="$(\Pi+D)/\epsilon$")
        ax.semilogx(k_plot, flux_K / eps, "r", label="$\Pi_K/\epsilon$")
        ax.semilogx(k_plot, flux_A / eps, "b", label="$\Pi_A/\epsilon$")
        ax.semilogx(k_plot, CK2A / eps, "m", linewidth=2, label="$B/\epsilon$")

        ax.legend()
