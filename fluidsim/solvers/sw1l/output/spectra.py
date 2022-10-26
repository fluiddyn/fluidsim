from typing import List, Optional
import functools
import h5py
import matplotlib as mpl
import numpy as np

from fluiddyn.util import mpi
from fluidsim.base.output.spectra import Spectra
from .normal_mode import NormalModeBase


class SpectraSW1L(Spectra):
    """Save and plot spectra."""

    def __init__(self, output):
        params = output.sim.params
        self.c2 = params.c2
        self.f = params.f

        super().__init__(output)

    def _init_online_plot(self):
        super()._init_online_plot()
        if mpi.rank == 0:
            self.ax.set_title("spectra\n" + self.output.summary_simul)

    def compute(self):
        """compute the values at one time."""
        # compute 'quantities_fft'
        (
            energyK_fft,
            energyA_fft,
            energyKr_fft,
        ) = self.output.compute_energies_fft()
        ErtelPE_fft, CharneyPE_fft = self.output.compute_PE_fft()

        # compute the spectra 1D
        spectrum1Dkx_EK, spectrum1Dky_EK = self.spectra1D_from_fft(energyK_fft)
        spectrum1Dkx_EA, spectrum1Dky_EA = self.spectra1D_from_fft(energyA_fft)
        spectrum1Dkx_EKr, spectrum1Dky_EKr = self.spectra1D_from_fft(energyKr_fft)
        spectrum1Dkx_EPE, spectrum1Dky_EPE = self.spectra1D_from_fft(ErtelPE_fft)
        spectrum1Dkx_CPE, spectrum1Dky_CPE = self.spectra1D_from_fft(
            CharneyPE_fft
        )

        dict_spectra1D = {
            "spectrum1Dkx_EK": spectrum1Dkx_EK,
            "spectrum1Dky_EK": spectrum1Dky_EK,
            "spectrum1Dkx_EA": spectrum1Dkx_EA,
            "spectrum1Dky_EA": spectrum1Dky_EA,
            "spectrum1Dkx_EKr": spectrum1Dkx_EKr,
            "spectrum1Dky_EKr": spectrum1Dky_EKr,
            "spectrum1Dkx_EPE": spectrum1Dkx_EPE,
            "spectrum1Dky_EPE": spectrum1Dky_EPE,
            "spectrum1Dkx_CPE": spectrum1Dkx_CPE,
            "spectrum1Dky_CPE": spectrum1Dky_CPE,
        }

        # compute the spectra 2D
        spectrum2D_EK = self.spectrum2D_from_fft(energyK_fft)
        spectrum2D_EA = self.spectrum2D_from_fft(energyA_fft)
        spectrum2D_EKr = self.spectrum2D_from_fft(energyKr_fft)
        spectrum2D_EPE = self.spectrum2D_from_fft(ErtelPE_fft)
        spectrum2D_CPE = self.spectrum2D_from_fft(CharneyPE_fft)

        dict_spectra2D = {
            "spectrum2D_EK": spectrum2D_EK,
            "spectrum2D_EA": spectrum2D_EA,
            "spectrum2D_EKr": spectrum2D_EKr,
            "spectrum2D_EPE": spectrum2D_EPE,
            "spectrum2D_CPE": spectrum2D_CPE,
        }

        dict_lin_spectra1D, dict_lin_spectra2D = self.compute_lin_spectra()
        dict_spectra1D.update(dict_lin_spectra1D)
        dict_spectra2D.update(dict_lin_spectra2D)

        return dict_spectra1D, dict_spectra2D

    def compute_lin_spectra(self):
        (
            energy_glin_fft,
            energy_dlin_fft,
            energy_alin_fft,
        ) = self.output.compute_lin_energies_fft()

        spectrum1Dkx_Eglin, spectrum1Dky_Eglin = self.spectra1D_from_fft(
            energy_glin_fft
        )
        spectrum1Dkx_Edlin, spectrum1Dky_Edlin = self.spectra1D_from_fft(
            energy_dlin_fft
        )
        spectrum1Dkx_Ealin, spectrum1Dky_Ealin = self.spectra1D_from_fft(
            energy_alin_fft
        )

        dict_spectra1D = {
            "spectrum1Dkx_Eglin": spectrum1Dkx_Eglin,
            "spectrum1Dky_Eglin": spectrum1Dky_Eglin,
            "spectrum1Dkx_Edlin": spectrum1Dkx_Edlin,
            "spectrum1Dky_Edlin": spectrum1Dky_Edlin,
            "spectrum1Dkx_Ealin": spectrum1Dkx_Ealin,
            "spectrum1Dky_Ealin": spectrum1Dky_Ealin,
        }

        spectrum2D_Eglin = self.spectrum2D_from_fft(energy_glin_fft)
        spectrum2D_Edlin = self.spectrum2D_from_fft(energy_dlin_fft)
        spectrum2D_Ealin = self.spectrum2D_from_fft(energy_alin_fft)

        dict_spectra2D = {
            "spectrum2D_Eglin": spectrum2D_Eglin,
            "spectrum2D_Edlin": spectrum2D_Edlin,
            "spectrum2D_Ealin": spectrum2D_Ealin,
        }

        return dict_spectra1D, dict_spectra2D

    def _online_plot_saving(self, dict_spectra1D, dict_spectra2D):
        if (
            self.params.oper.nx == self.params.oper.ny
            and self.params.oper.Lx == self.params.oper.Ly
        ):
            spectrum2D_EK = dict_spectra2D["spectrum2D_EK"]
            spectrum2D_EA = dict_spectra2D["spectrum2D_EA"]
            spectrum2D_EKr = dict_spectra2D["spectrum2D_EKr"]
            spectrum2D_E = spectrum2D_EK + spectrum2D_EA
            spectrum2D_EKd = spectrum2D_EK - spectrum2D_EKr
            khE = self.oper.khE
            coef_norm = khE ** (3.0)
            self.ax.loglog(khE, spectrum2D_E * coef_norm, "k")
            self.ax.loglog(khE, spectrum2D_EK * coef_norm, "r")
            self.ax.loglog(khE, spectrum2D_EA * coef_norm, "b")
            self.ax.loglog(khE, spectrum2D_EKr * coef_norm, "r--")
            self.ax.loglog(khE, spectrum2D_EKd * coef_norm, "r:")
            lin_inf, lin_sup = self.ax.get_ylim()
            if lin_inf < 10e-6:
                lin_inf = 10e-6
            self.ax.set_ylim([lin_inf, lin_sup])
        else:
            print(
                "you need to implement the ploting "
                "of the spectra for this case"
            )

    def plot1d(
        self,
        tmin: float = 0,
        tmax: float = 1000,
        delta_t: float = 2,
        coef_compensate: float = 3,
        coef_norm: Optional[np.ndarray] = None,
        ax: Optional[mpl.axes.Axes] = None,
        help_lines: bool = True,
    ):

        with h5py.File(self.path_file1D, "r") as h5file:
            dset_times = h5file["times"]
            times = dset_times[...]
            # nb_spectra = times.shape[0]

            dset_kxE = h5file["kxE"]
            # dset_kyE = h5file['kyE']

            kh = dset_kxE[...]

            dset_spectrum1Dkx_EK = h5file["spectrum1Dkx_EK"]
            dset_spectrum1Dky_EK = h5file["spectrum1Dky_EK"]
            dset_spectrum1Dkx_EA = h5file["spectrum1Dkx_EA"]
            dset_spectrum1Dky_EA = h5file["spectrum1Dky_EA"]

            # dset_spectrum1Dkx_EKr = h5file["spectrum1Dkx_EKr"]
            # dset_spectrum1Dky_EKr = h5file["spectrum1Dky_EKr"]

            # nt = len(times)

            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))
            if delta_i_plot == 0 and delta_t != 0.0:
                delta_i_plot = 1
            delta_t = delta_i_plot * delta_t_save

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            to_print = "plot1d(tmin={}, tmax={}, delta_t={:.2f},".format(
                tmin, tmax, delta_t
            )
            print(to_print)

            to_print = """plot 1D spectra
    tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
    imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
                tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
            )
            print(to_print)

            if ax is None:
                fig, ax = self.output.figure_axe()

            ax.set_xlabel("$k_h$")
            ax.set_ylabel("1D spectra")
            ax.set_title("1D spectra\n" + self.output.summary_simul)
            ax.set_xscale("log")
            ax.set_yscale("log")

            if coef_norm is None:
                coef_norm = kh ** (coef_compensate)

            # min_to_plot = 1e-16

            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot + 1, delta_i_plot):
                    E_K = dset_spectrum1Dkx_EK[it] + dset_spectrum1Dky_EK[it]
                    # E_K[E_K<min_to_plot] = 0.
                    E_A = dset_spectrum1Dkx_EA[it] + dset_spectrum1Dky_EA[it]
                    # E_A[E_A<min_to_plot] = 0.
                    E_tot = E_K + E_A

                    # E_Kr = dset_spectrum1Dkx_EKr[it] + dset_spectrum1Dky_EKr[it]
                    # E_Kr[E_Kr<min_to_plot] = 0.
                    # E_Kd = E_K - E_Kr

                    ax.plot(kh, E_tot * coef_norm, "k", linewidth=2)
                    ax.plot(kh, E_K * coef_norm, "r", linewidth=1)
                    ax.plot(kh, E_A * coef_norm, "b", linewidth=1)
            # ax.plot(kh, E_Kr*coef_norm, 'r--', linewidth=1)
            # ax.plot(kh, E_Kd*coef_norm, 'r:', linewidth=1)

            E_K = (
                dset_spectrum1Dkx_EK[imin_plot : imax_plot + 1]
                + dset_spectrum1Dky_EK[imin_plot : imax_plot + 1]
            ).mean(0)

            E_A = (
                dset_spectrum1Dkx_EA[imin_plot : imax_plot + 1]
                + dset_spectrum1Dky_EA[imin_plot : imax_plot + 1]
            ).mean(0)

        ax.plot(kh, E_K * coef_norm, "r", linewidth=2)
        ax.plot(kh, E_A * coef_norm, "b", linewidth=2)

        if help_lines:
            kh_pos = kh[kh > 0]
            coef_norm = coef_norm[kh > 0]
            ax.plot(kh_pos, kh_pos ** (-3) * coef_norm, "k--", linewidth=1)
            ax.plot(kh_pos, kh_pos ** (-5.0 / 3) * coef_norm, "k-.", linewidth=1)

    def plot2d(
        self,
        tmin: float = 0,
        tmax: float = 1000,
        delta_t: float = 2,
        coef_compensate: float = 3,
        coef_norm: Optional[np.ndarray] = None,
        keys: List[str] = ["Etot", "EK", "EA", "EKr", "EKd"],
        colors: List[str] = ["k", "r", "b", "r--", "r:"],
        kh_norm: float = 1,
        ax: Optional[mpl.axes.Axes] = None,
        help_lines: bool = True,
    ):

        with h5py.File(self.path_file2D, "r") as h5file:
            times = h5file["times"][...]

            dset_khE = h5file["khE"]
            kh = dset_khE[...] / kh_norm

            dset_spectrumEK = h5file["spectrum2D_EK"]
            dset_spectrumEA = h5file["spectrum2D_EA"]
            dset_spectrumEKr = h5file["spectrum2D_EKr"]

            delta_t_save = np.mean(times[1:] - times[0:-1])
            delta_i_plot = int(np.round(delta_t / delta_t_save))
            if delta_i_plot == 0 and delta_t != 0.0:
                delta_i_plot = 1
            delta_t = delta_i_plot * delta_t_save

            imin_plot = np.argmin(abs(times - tmin))
            imax_plot = np.argmin(abs(times - tmax))

            tmin_plot = times[imin_plot]
            tmax_plot = times[imax_plot]

            to_print = "plot2d(tmin={}, tmax={}, delta_t={:.2f},".format(
                tmin, tmax, delta_t
            )
            print(to_print)

            to_print = """plot 2D spectra
    tmin = {:8.6g} ; tmax = {:8.6g} ; delta_t = {:8.6g}
    imin = {:8d} ; imax = {:8d} ; delta_i = {:8d}""".format(
                tmin_plot, tmax_plot, delta_t, imin_plot, imax_plot, delta_i_plot
            )
            print(to_print)

            if ax is None:
                fig, ax = self.output.figure_axe()

            ax.set_xlabel("$k_h$")
            ax.set_ylabel("2D spectra")
            ax.set_title("2D spectra\n" + self.output.summary_simul)
            ax.set_xscale("log")
            ax.set_yscale("log")

            if coef_norm is None:
                coef_norm = kh**coef_compensate

            machine_zero = 1e-15
            if delta_t != 0.0:
                for it in range(imin_plot, imax_plot + 1, delta_i_plot):
                    for k, c in zip(keys, colors):
                        dset = self._get_field_to_plot(it, k, h5file)
                        dset[dset < 10e-16] = machine_zero
                        ax.plot(kh, dset * coef_norm, c, linewidth=1)

            EK = dset_spectrumEK[imin_plot : imax_plot + 1].mean(0)
            EA = dset_spectrumEA[imin_plot : imax_plot + 1].mean(0)
            EKr = dset_spectrumEKr[imin_plot : imax_plot + 1].mean(0)

            EK[abs(EK) < 10e-16] = machine_zero
            EA[abs(EA) < 10e-16] = machine_zero
            EKr[abs(EKr) < 10e-16] = machine_zero

            E_tot = EK + EA
            EKd = EK - EKr + machine_zero

            if "Etot" in keys:
                ax.plot(
                    kh,
                    E_tot * coef_norm,
                    colors[0],
                    linewidth=2,
                    label="$E_{tot}$",
                )

            if "EK" in keys:
                ax.plot(kh, EK * coef_norm, "r", linewidth=1, label="$E_{K}$")
                ax.plot(kh, -EK * coef_norm, "k-", linewidth=1)

            if "EA" in keys:
                ax.plot(kh, EA * coef_norm, "b", linewidth=1, label="$E_{A}$")

            if "EKr" in keys:
                ax.plot(kh, EKr * coef_norm, "r--", linewidth=1, label="$E_{Kr}$")

            if "EKd" in keys:
                ax.plot(kh, EKd * coef_norm, "r:", linewidth=1, label="$E_{Kd}$")
                ax.plot(kh, -EKd * coef_norm, "k:", linewidth=1)

            self._plot2d_lin_spectra(
                h5file, ax, imin_plot, imax_plot, kh, coef_norm, keys
            )

        if help_lines:
            kh_pos = kh[kh > 0]
            coef_norm = coef_norm[kh > 0]
            ax.plot(kh_pos, kh_pos ** (-2) * coef_norm, "k-", linewidth=1)
            ax.plot(kh_pos, kh_pos ** (-3) * coef_norm, "k--", linewidth=1)
            ax.plot(kh_pos, kh_pos ** (-5.0 / 3) * coef_norm, "k-.", linewidth=1)

            postxt = kh.max()
            ax.text(postxt, postxt ** (-2 + coef_compensate), r"$k^{-2}$")
            ax.text(postxt, postxt ** (-3 + coef_compensate), r"$k^{-3}$")
            ax.text(postxt, postxt ** (-5.0 / 3 + coef_compensate), r"$k^{-5/3}$")

        ax.legend()

    def plot_diss(
        self,
        tmin=0,
        tmax=1000,
        delta_t=2,
        keys=["Dtot", "DK", "DA", "DKr", "DKd"],
        colors=["k", "r", "b", "r--", "r:"],
        kh_norm=1,
        ax=None,
    ):
        """Plot the dissipation spectra."""

        def get_nu(o):
            return getattr(self.sim.params, f"nu_{o}")

        for order in [2, 4, 8]:
            nu = get_nu(order)
            if nu > 0:
                break
        else:
            raise ValueError("Viscosity is zero?")

        with h5py.File(self.path_file2D, "r") as h5file:
            dset_khE = h5file["khE"]
            kh = dset_khE[...]

        coef_norm = 2 * nu * kh**order
        keys = ["E" + k.lstrip("D") for k in keys]
        self.plot2d(
            tmin, tmax, delta_t, 0, coef_norm, keys, colors, kh_norm, ax, False
        )

    def _plot2d_lin_spectra(
        self, h5file, ax, imin_plot, imax_plot, kh, coef_norm, keys
    ):
        machine_zero = 1e-15
        if self.sim.info.solver.short_name.startswith("SW1L"):
            dset_spectrumEdlin = h5file["spectrum2D_Edlin"]
            Edlin = (
                dset_spectrumEdlin[imin_plot : imax_plot + 1].mean(0)
                + machine_zero
            )
            ax.plot(kh, Edlin * coef_norm, "c", linewidth=1, label="$E_{D}$")

        if self.params.f != 0:
            dset_spectrumEglin = h5file["spectrum2D_Eglin"]
            Eglin = (
                dset_spectrumEglin[imin_plot : imax_plot + 1].mean(0)
                + machine_zero
            )
            ax.plot(kh, Eglin * coef_norm, "g", linewidth=1, label="$E_{G}$")

            dset_spectrumEalin = h5file["spectrum2D_Ealin"]
            Ealin = (
                dset_spectrumEalin[imin_plot : imax_plot + 1].mean(0)
                + machine_zero
            )
            ax.plot(kh, Ealin * coef_norm, "y", linewidth=1, label="$E_{A}$")

    def _get_field_to_plot(self, idx, key_field=None, h5file=None):
        if key_field is None:
            key_field = self._ani_key

        def select(idx, key_field, h5file):
            if key_field == "Etot" or key_field is None:
                self._ani_key = "Etot"
                y = h5file["spectrum2D_EK"][idx] + h5file["spectrum2D_EA"][idx]
            elif key_field == "EKd":
                y = h5file["spectrum2D_EK"][idx] - h5file["spectrum2D_EKr"][idx]
            else:
                try:
                    key_field = "spectrum2D_" + key_field
                    y = h5file[key_field][idx]
                except:
                    raise KeyError("Unknown key ", key_field)

            return y

        if h5file is None:
            with h5py.File(self.path_file2D) as h5file:
                return select(idx, key_field, h5file)

        else:
            return select(idx, key_field, h5file)


class SpectraSW1LNormalMode(SpectraSW1L):
    def __init__(self, output):
        self.norm_mode = NormalModeBase(output)
        super().__init__(output)

    def compute_lin_spectra(self):
        (
            energy_glin_fft,
            energy_aplin_fft,
            energy_amlin_fft,
        ) = self.norm_mode.compute_qapam_energies_fft()

        energy_alin_fft = energy_aplin_fft + energy_amlin_fft
        spectrum1Dkx_Eglin, spectrum1Dky_Eglin = self.spectra1D_from_fft(
            energy_glin_fft
        )
        spectrum1Dkx_Ealin, spectrum1Dky_Ealin = self.spectra1D_from_fft(
            energy_alin_fft
        )

        dict_spectra1D = {
            "spectrum1Dkx_Eglin": spectrum1Dkx_Eglin,
            "spectrum1Dky_Eglin": spectrum1Dky_Eglin,
            "spectrum1Dkx_Ealin": spectrum1Dkx_Ealin,
            "spectrum1Dky_Ealin": spectrum1Dky_Ealin,
        }

        spectrum2D_Eglin = self.spectrum2D_from_fft(energy_glin_fft)
        spectrum2D_Ealin = self.spectrum2D_from_fft(energy_alin_fft)

        dict_spectra2D = {
            "spectrum2D_Eglin": spectrum2D_Eglin,
            "spectrum2D_Ealin": spectrum2D_Ealin,
        }

        return dict_spectra1D, dict_spectra2D

    def plot2d(
        self,
        tmin=0,
        tmax=1000,
        delta_t=2,
        coef_compensate=3,
        coef_norm=None,
        keys=["Etot", "EK", "Eglin", "Ealin"],
        colors=["k", "r", "g", "y"],
        kh_norm=1,
        ax=None,
        help_lines=True,
    ):
        # Ideally functool.partialmethod would suffice, but issues due to mixing args and kwargs
        super().plot2d(
            tmin,
            tmax,
            delta_t,
            coef_compensate,
            coef_norm,
            keys,
            colors,
            kh_norm,
            ax,
            help_lines,
        )

    def _plot2d_lin_spectra(
        self, h5file, ax, imin_plot, imax_plot, kh, coef_norm, keys
    ):
        machine_zero = 1e-15
        if self.sim.info.solver.short_name.startswith("SW1L"):
            if "spectrum2D_Edlin" in h5file.keys():
                dset_spectrumEalin = h5file[
                    "spectrum2D_Edlin"
                ]  # TODO: To be removed. Kept for compatibility
            else:
                dset_spectrumEalin = h5file["spectrum2D_Ealin"]

            if "Ealin" in keys:
                Ealin = (
                    dset_spectrumEalin[imin_plot : imax_plot + 1].mean(0)
                    + machine_zero
                )
                ax.plot(
                    kh, Ealin * coef_norm, "y", linewidth=1, label="$E_{AGEO}$"
                )

            if "Eglin" in keys:
                dset_spectrumEglin = h5file["spectrum2D_Eglin"]
                Eglin = (
                    dset_spectrumEglin[imin_plot : imax_plot + 1].mean(0)
                    + machine_zero
                )
                ax.plot(
                    kh, Eglin * coef_norm, "g", linewidth=1, label="$E_{GEO}$"
                )
