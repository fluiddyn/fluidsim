"""SEB (:mod:`fluidsim.solvers.ns3d.strat.output.spect_energy_budget`)
======================================================================

.. autoclass:: SpectralEnergyBudgetNS3D
   :members:
   :private-members:

"""

from textwrap import dedent

import numpy as np
import h5py

from fluiddyn.util import mpi
from fluidfft.fft3d.operators import vector_product

from fluidsim.base.output.base import SpecificOutput


class SpectralEnergyBudgetNS3D(SpecificOutput):
    r"""Spectral energy budget of ns3d.strat.

    .. |vv| mathmacro:: \textbf{v}

    .. |kk| mathmacro:: \textbf{k}

    .. |convK2A| mathmacro:: C_{K\rightarrow A}

    .. |bnabla| mathmacro:: \boldsymbol{\nabla}

    .. |bomega| mathmacro:: \boldsymbol{\omega}

    Notes
    -----

    .. math::

      d_t E_K(\kk) = T_K(\kk) - D_K(\kk),

    where :math:`E_K(\kk) = |\hat{\vv}|^2/2`.

    The transfer term is

    .. math::

      T_K(\kk) = \Re \left(\hat{\vv}^* \cdot P_\perp\widehat{\vv \times \bomega} \right).

    By definition, we have to have :math:`\sum T_K(\kk) = 0`.

    The dissipative term is

    .. math::

      D_K(\kk) = f_{dK}(\kk) E_K(\kk),

    where :math:`f_{dK}(\kk)` is the dissipation frequency depending of the
    wavenumber and the viscous coefficients.

    """

    _tag = "spect_energy_budg"
    _name_file = _tag + ".h5"

    @classmethod
    def _complete_params_with_default(cls, params):
        tag = cls._tag

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag, attribs={"HAS_TO_PLOT_SAVED": False})
        params.output.spectra._set_doc(
            dedent(
                """
                    HAS_TO_PLOT_SAVED : bool (False)

                      If True, some curves can be plotted during the run.
        """
            )
        )

    def __init__(self, output):

        params = output.sim.params
        self.nx = params.oper.nx

        self.oper = oper = output.sim.oper

        kx = oper.deltakx * np.arange(oper.nkx_spectra)
        ky = oper.deltaky * np.arange(oper.nky_spectra)
        kz = oper.deltakz * np.arange(oper.nkz_spectra)

        super().__init__(
            output,
            period_save=params.output.periods_save.spect_energy_budg,
            has_to_plot_saved=params.output.spect_energy_budg.HAS_TO_PLOT_SAVED,
            arrays_1st_time={"kx": kx, "ky": ky, "kz": kz, "kh": oper.kh_spectra},
        )

    def compute_spectra(self, name, quantity):
        spectrum_kzkh = self.oper.compute_spectrum_kzkh(quantity)
        spectrum_kx, spectrum_ky, spectrum_kz = self.oper.compute_1dspectra(
            quantity
        )
        return {
            name: spectrum_kzkh,
            name + "_kx": spectrum_kx,
            name + "_ky": spectrum_ky,
            name + "_kz": spectrum_kz,
        }

    def compute(self):

        results = {}
        state = self.sim.state

        oper = self.sim.oper
        ifft_as_arg_destroy = oper.ifft_as_arg_destroy

        state_spect = state.state_spect

        vx_fft = state_spect.get_var("vx_fft")
        vy_fft = state_spect.get_var("vy_fft")
        vz_fft = state_spect.get_var("vz_fft")

        omegax_fft, omegay_fft, omegaz_fft = oper.rotfft_from_vecfft(
            vx_fft, vy_fft, vz_fft
        )

        omegax = state.fields_tmp[3]
        omegay = state.fields_tmp[4]
        omegaz = state.fields_tmp[5]

        ifft_as_arg_destroy(omegax_fft, omegax)
        ifft_as_arg_destroy(omegay_fft, omegay)
        ifft_as_arg_destroy(omegaz_fft, omegaz)

        state_phys = state.state_phys
        vx = state_phys.get_var("vx")
        vy = state_phys.get_var("vy")
        vz = state_phys.get_var("vz")

        fx, fy, fz = vector_product(vx, vy, vz, omegax, omegay, omegaz)

        fx_fft = oper.fft(fx)
        fy_fft = oper.fft(fy)
        fz_fft = oper.fft(fz)

        oper.project_perpk3d(fx_fft, fy_fft, fz_fft)

        results.update(
            self.compute_spectra(
                "transfer_Kh",
                np.real(vx_fft.conj() * fx_fft) + np.real(vy_fft.conj() * fy_fft),
            )
        )

        results.update(
            self.compute_spectra("transfer_Kz", np.real(vz_fft.conj() * fz_fft))
        )

        urx_fft, ury_fft, _, _ = self.sim.oper.urudfft_from_vxvyfft(
            vx_fft, vy_fft
        )

        results.update(
            self.compute_spectra(
                "transfer_Khr",
                np.real(urx_fft.conj() * fx_fft)
                + np.real(ury_fft.conj() * fy_fft),
            )
        )

        del urx_fft, ury_fft, _

        f_d, f_d_hypo = self.sim.compute_freq_diss()
        # todo: f_d_hypo

        results.update(
            self.compute_spectra(
                "diss_Kh", f_d * (abs(vx_fft) ** 2 + abs(vy_fft) ** 2)
            )
        )

        del vx_fft, vy_fft

        results.update(self.compute_spectra("diss_Kz", f_d * abs(vz_fft) ** 2))

        transfer_K = results["transfer_Kh"] + results["transfer_Kz"]
        assert transfer_K.sum() < 1e-14

        return results

    def load_mean(self, tmin=0, tmax=None, key_to_load=None):
        means = {}
        with h5py.File(self.path_file, "r") as file:
            times = file["times"][...]
            nt = len(times)
            if tmin is None:
                imin_plot = 0
            else:
                imin_plot = np.argmin(abs(times - tmin))
            if tmax is None:
                imax_plot = nt - 1
            else:
                imax_plot = np.argmin(abs(times - tmax))

            tmin = times[imin_plot]
            tmax = times[imax_plot]

            print(
                "compute mean spectral energy budget\n"
                f"tmin = {tmin:8.6g} ; tmax = {tmax:8.6g}\n"
                f"imin = {imin_plot:8d} ; imax = {imax_plot:8d}"
            )

            for key in list(file.keys()):
                if key.startswith("k"):
                    means[key] = file[key][...]

            if key_to_load is not None:
                if key_to_load not in file.keys():
                    print(key_to_load, file.keys())
                    raise ValueError
                spect = file[key_to_load][imin_plot : imax_plot + 1].mean(0)
                means[key_to_load] = spect
                return means

            for key in list(file.keys()):
                if key != "times" and not key.startswith("k"):
                    dset_key = file[key]
                    spect = dset_key[imin_plot : imax_plot + 1].mean(0)
                    means[key] = spect
        return means

    _key_plot_default_kzkh = "transfer_Kh"

    def plot_kzkh(self, tmin=0, tmax=None, key=None, ax=None):

        with h5py.File(self.path_file, "r") as file:
            keys_saved = [
                key
                for key in file.keys()
                if key not in ("times", "info_simul")
                and not key.startswith("k")
                and not any(key.endswith("_k" + letter) for letter in "xyz")
            ]

        if key is None:
            key = self._key_plot_default_kzkh

        if key not in keys_saved:
            raise ValueError(f"key '{key}' not in {keys_saved}")

        data = self.load_mean(tmin, tmax, key)
        spectrum = data[key]
        kz = data["kz"]
        kh = data["kh"]

        if ax is None:
            fig, ax = self.output.figure_axe()

        ax.set_xlabel("$\kappa_h$")
        ax.set_ylabel("$k_z$")
        ax.set_title(
            f"{key}, solver {self.output.name_solver}, nx = {self.nx:5d}"
        )

        ax.pcolormesh(kh, kz, spectrum)

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
        DKh = data[key_flux + "diss_Kh"]
        DKz = data[key_flux + "diss_Kz"]

        flux_tot = flux_Kh + flux_Kz
        D = DKh + DKz

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

        ax.semilogx(
            k_plot, flux_tot / eps, "k", linewidth=2, label="$\Pi/\epsilon$"
        )
        ax.semilogx(k_plot, D / eps, "k--", linewidth=2, label="$D/\epsilon$")
        ax.semilogx(
            k_plot, (flux_tot + D) / eps, "k:", label="$(\Pi+D)/\epsilon$"
        )

        ax.legend()
