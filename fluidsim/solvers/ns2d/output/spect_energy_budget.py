"""Energy budget (:mod:`fluidsim.solvers.ns2d.output.spect_energy_budget`)
==========================================================================

.. autoclass:: SpectralEnergyBudgetNS2D
   :members:
   :private-members:

"""

import numpy as np
import h5py


from fluidsim.base.output.spect_energy_budget import (
    SpectralEnergyBudgetBase,
    cumsum_inv,
)


class SpectralEnergyBudgetNS2D(SpectralEnergyBudgetBase):
    r"""Save and plot the spectral energy and enstrophy budgets.

    Notes
    -----

    .. math::

      d_t E(k_h) = T_E(k_h) - D_E(k_h),

      d_t Z(k_h) = T_Z(k_h) - D_Z(k_h),

    where :math:`E(k_h)` and :math:`Z(k_h)` are the energy and enstrophy
    spectra. The transfer terms are

    .. math::

      T_E(\mathbf{k}) = \Re (\hat{u}_i^* \widehat{N_i}),
      \quad
      T_Z(\mathbf{k}) = \Re (\hat{\zeta}^* \widehat{N_\zeta}),

    with :math:`N_i = -u_j \partial_j u_i` and
    :math:`N_\zeta = -u_j \partial_j \zeta - \beta u_y`. Both the
    non-linear terms and the :math:`\beta` term conserve :math:`E` and
    :math:`Z`, so :math:`\sum T_E = \sum T_Z = 0`.

    Only the transfers are saved. The fluxes are obtained by integrating
    them from the large wavenumbers,

    .. math:: \Pi(k_h) = \sum_{k_h' \geq k_h} T(k_h') \delta k,

    whereas the cumulated dissipation is integrated from the small ones,

    .. math:: D(k_h) = \sum_{k_h' < k_h} 2 f_d(k_h') E(k_h') \delta k,

    where :math:`f_d` is the dissipation frequency, recomputed from the
    `params.nu_...` parameters. In 2d it only depends on :math:`|k|`, so
    it is constant over a shell and :math:`D` does not need to be saved.
    See :func:`compute_fluxes_mean` for the accuracy of this
    reconstruction.

    For a statistically steady forced simulation, :math:`\Pi + D` should
    be equal to the cumulated injection.

    """

    def compute(self):
        """Compute the spectral energy and enstrophy transfers at one time."""
        oper = self.sim.oper

        ux = self.sim.state.state_phys.get_var("ux")
        uy = self.sim.state.state_phys.get_var("uy")

        rot_fft = self.sim.state.state_spect.get_var("rot_fft")
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)

        px_rot_fft, py_rot_fft = oper.gradfft_from_fft(rot_fft)
        px_rot = oper.ifft2(px_rot_fft)
        py_rot = oper.ifft2(py_rot_fft)

        px_ux_fft, py_ux_fft = oper.gradfft_from_fft(ux_fft)
        px_ux = oper.ifft2(px_ux_fft)
        py_ux = oper.ifft2(py_ux_fft)

        px_uy_fft, py_uy_fft = oper.gradfft_from_fft(uy_fft)
        px_uy = oper.ifft2(px_uy_fft)
        py_uy = oper.ifft2(py_uy_fft)

        Frot = -ux * px_rot - uy * (py_rot + self.params.beta)
        Frot_fft = oper.fft2(Frot)
        oper.dealiasing(Frot_fft)

        Fx = -ux * px_ux - uy * (py_ux)
        Fx_fft = oper.fft2(Fx)
        oper.dealiasing(Fx_fft)

        Fy = -ux * px_uy - uy * (py_uy)
        Fy_fft = oper.fft2(Fy)
        oper.dealiasing(Fy_fft)

        transferZ_fft = (
            np.real(rot_fft.conj() * Frot_fft + rot_fft * Frot_fft.conj()) / 2.0
        )
        # print ('sum(transferZ) = {0:9.4e} ; sum(abs(transferZ)) = {1:9.4e}'
        #       ).format(self.sum_wavenumbers(transferZ_fft),
        #                self.sum_wavenumbers(abs(transferZ_fft)))

        transferE_fft = (
            np.real(
                ux_fft.conj() * Fx_fft
                + ux_fft * Fx_fft.conj()
                + uy_fft.conj() * Fy_fft
                + uy_fft * Fy_fft.conj()
            )
            / 2.0
        )
        # print ('sum(transferE) = {0:9.4e} ; sum(abs(transferE)) = {1:9.4e}'
        #       ).format(self.sum_wavenumbers(transferE_fft),
        #                self.sum_wavenumbers(abs(transferE_fft)))

        transfer2D_E = self.spectrum2D_from_fft(transferE_fft)
        transfer2D_Z = self.spectrum2D_from_fft(transferZ_fft)

        dict_results = {
            "transfer2D_E": transfer2D_E,
            "transfer2D_Z": transfer2D_Z,
        }
        return dict_results

    def _online_plot_saving(self, dict_results):
        transfer2D_E = dict_results["transfer2D_E"]
        transfer2D_Z = dict_results["transfer2D_Z"]
        khE = self.oper.khE
        PiE = cumsum_inv(transfer2D_E) * self.oper.deltak
        PiZ = cumsum_inv(transfer2D_Z) * self.oper.deltak
        self.axe_a.plot(khE + khE[1], PiE, "k")
        self.axe_b.plot(khE + khE[1], PiZ, "g")

    def load_mean(self, tmin=0, tmax=None, keys_to_load=None, verbose=True):
        """Load the spectra averaged between tmin and tmax."""
        means = {}
        with h5py.File(self.path_file, "r") as file:
            times = file["times"][...]
            nt = len(times)

            imin = 0 if tmin is None else np.argmin(abs(times - tmin))
            imax = nt - 1 if tmax is None else np.argmin(abs(times - tmax))

            if verbose:
                print(
                    "compute mean spectral energy budget\n"
                    f"tmin = {times[imin]:8.6g} ; tmax = {times[imax]:8.6g}\n"
                    f"imin = {imin:8d} ; imax = {imax:8d}"
                )

            for key in file.keys():
                if key.startswith("kh"):
                    means[key] = file[key][...]

            keys_saved = [
                key
                for key in file.keys()
                if key != "times" and not key.startswith(("k", "info"))
            ]

            if keys_to_load is None:
                keys_to_load = keys_saved
            else:
                if isinstance(keys_to_load, str):
                    keys_to_load = [keys_to_load]
                for key in keys_to_load:
                    if key not in keys_saved:
                        raise ValueError(f"key '{key}' not in {keys_saved}")

            for key in keys_to_load:
                means[key] = file[key][imin : imax + 1].mean(0)

        return means

    def _freq_diss_kh(self, kh):
        params = self.params
        f_d = np.zeros_like(kh)
        for order in (2, 4, 8):
            nu = getattr(params, f"nu_{order}", 0.0)
            if nu:
                f_d = f_d + nu * kh**order
        nu_m4 = getattr(params, "nu_m4", 0.0)
        if nu_m4:
            kh_not0 = np.where(kh == 0, np.inf, kh)
            f_d = f_d + nu_m4 * kh_not0**-4
        return f_d

    def _load_spectra2d_mean(self, tmin=0, tmax=None):
        with h5py.File(self.output.spectra.path_file2D, "r") as file:
            times = file["times"][...]
            imin = 0 if tmin is None else np.argmin(abs(times - tmin))
            imax = (
                len(times) - 1 if tmax is None else np.argmin(abs(times - tmax))
            )
            kh = file["khE"][...]
            E = file["spectrum2D_E"][imin : imax + 1].mean(0)
        return kh, E

    def compute_fluxes_mean(self, tmin=0, tmax=None, verbose=False):
        """Compute the mean fluxes and cumulated dissipations.

        The dissipations are reconstructed from the 2d spectra and the
        dissipation frequency evaluated at the center of each shell. The
        resulting D[-1] overestimates the dissipation rate given by
        spatial_means by a few percents, more for high order
        viscosities. The keys "DE" and "DZ" are absent if the 2d spectra
        were not saved.

        """
        data = self.load_mean(tmin, tmax, verbose=verbose)

        khE = data["khE"]
        deltak = khE[1] - khE[0]

        results = {
            "khE": khE,
            "PiE": deltak * cumsum_inv(data["transfer2D_E"]),
            "PiZ": deltak * cumsum_inv(data["transfer2D_Z"]),
        }

        try:
            kh, E = self._load_spectra2d_mean(tmin, tmax)
        except (OSError, KeyError):
            pass
        else:
            f_d = self._freq_diss_kh(kh)
            results["DE"] = deltak * np.cumsum(2 * f_d * E)
            results["DZ"] = deltak * np.cumsum(2 * f_d * kh**2 * E)

        return results

    def plot_fluxes(self, tmin=0, tmax=None, key="both", normalize=True, ax=None):
        """Plot the mean spectral fluxes.

        Parameters
        ----------

        key : {"both", "E", "Z"}

          Plot the energy budget, the enstrophy budget, or both, each
          normalized by its own dissipation rate.

        normalize : bool

          Normalize by D[-1].

        """
        data = self.compute_fluxes_mean(tmin, tmax)

        khE = data["khE"]
        k_plot = khE + (khE[1] - khE[0]) / 2

        if ax is None:
            fig, ax = self.output.figure_axe()

        keys = ["E", "Z"] if key == "both" else [key]
        colors = {"E": "k", "Z": "g"}

        for key_ in keys:
            Pi = data["Pi" + key_]
            D = data.get("D" + key_)
            eps = D[-1] if (normalize and D is not None) else 1.0

            color = colors[key_]
            ax.semilogx(
                k_plot, Pi / eps, color, linewidth=2, label=r"$\Pi_" + key_ + "$"
            )
            if D is not None:
                ax.semilogx(
                    k_plot,
                    D / eps,
                    color + "--",
                    linewidth=2,
                    label="$D_" + key_ + "$",
                )
                ax.semilogx(
                    k_plot,
                    (Pi + D) / eps,
                    color + ":",
                    label=r"$\Pi_" + key_ + " + D_" + key_ + "$",
                )

        ax.set_ylabel(r"$\Pi(k_h) / \epsilon$" if normalize else r"$\Pi(k_h)$")
        ax.axhline(0, color="0.7", linewidth=0.5)
        ax.set_xlabel("$k_h$")
        ax.set_title(f"spectral fluxes\n{self.output.summary_simul}")
        ax.legend()

        return ax

    def plot(self, tmin=0, tmax=1000, delta_t=2):
        with h5py.File(self.path_file, "r") as h5file:
            dset_times = h5file["times"]
            dset_khE = h5file["khE"]
            khE = dset_khE[...]
            khE = khE + khE[1]

            dset_transferE = h5file["transfer2D_E"]
            dset_transferZ = h5file["transfer2D_Z"]

            # nb_spectra = dset_times.shape[0]
            times = dset_times[...]
            # nt = len(times)

            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))

            if delta_i_plot == 0 and delta_t != 0.0:
                delta_i_plot = 1
            delta_t = delta_i_plot * delta_t_save

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            print(f"plot(tmin={tmin}, tmax={tmax}, delta_t={delta_t:.2f})")

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]
            print(
                f"""plot spectral energy budget
    tmin = {tmin_plot:8.6g} ; tmax = {tmax_plot:8.6g} ; delta_t = {delta_t:8.6g}
    imin = {imin_plot:8d} ; imax = {imax_plot:8d} ; delta_i = {delta_i_plot:8d}"""
            )

            fig, ax1 = self.output.figure_axe()
            ax1.set_xlabel("$k_h$")
            ax1.set_ylabel(r"$\Pi(k_h)$")
            ax1.set_xscale("log")
            ax1.set_yscale("linear")
            ax1.axhline(0, color="0.7", linewidth=0.5)
            ax1.set_title(
                f"spectral fluxes, {imax_plot - imin_plot + 1} times\n"
                f"{self.output.summary_simul}"
            )

            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot, delta_i_plot):
                    transferE = dset_transferE[it]
                    transferZ = dset_transferZ[it]

                    PiE = cumsum_inv(transferE) * self.oper.deltak
                    PiZ = cumsum_inv(transferZ) * self.oper.deltak

                    ax1.plot(khE, PiE, color="0.8", linewidth=0.5)
                    ax1.plot(khE, PiZ, color="0.8", linewidth=0.5)

            transferE = dset_transferE[imin_plot:imax_plot].mean(0)
            transferZ = dset_transferZ[imin_plot:imax_plot].mean(0)

        PiE = cumsum_inv(transferE) * self.oper.deltak
        PiZ = cumsum_inv(transferZ) * self.oper.deltak

        ax1.plot(khE, PiE, "k", linewidth=2, label=r"$\Pi_E$")
        ax1.plot(khE, PiZ, "g", linewidth=2, label=r"$\Pi_Z$")
        ax1.legend()
