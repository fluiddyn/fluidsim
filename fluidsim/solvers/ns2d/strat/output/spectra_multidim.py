"""Multidimensional spectra output (:mod:`fluidsim.solvers.ns2d.strat.output.spectra_multidim`)
===============================================================================================

.. autoclass:: SpectraMultiDimNS2DStrat
   :members:
   :private-members:

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from math import radians
from fluidsim.base.output.spectra_multidim import SpectraMultiDim


class SpectraMultiDimNS2DStrat(SpectraMultiDim):
    """Save and plot the spectra."""

    def compute(self):
        """Computes multidimensional spectra at one time."""

        # Get variables
        energyK_fft, energyA_fft = self.output.compute_energies_fft()

        ap_fft = self.sim.state.compute("ap_fft")
        am_fft = self.sim.state.compute("am_fft")

        # Computes multidimensional spectra
        spectrumkykx_EK = self.oper.compute_spectrum_kykx(
            energyK_fft, folded=False
        )
        spectrumkykx_EA = self.oper.compute_spectrum_kykx(
            energyA_fft, folded=False
        )

        energy_ap_fft = abs(ap_fft) ** 2
        energy_am_fft = abs(am_fft) ** 2

        spectrumkykx_ap = self.oper.compute_spectrum_kykx(
            energy_ap_fft, folded=False
        )
        spectrumkykx_am = self.oper.compute_spectrum_kykx(
            energy_am_fft, folded=False
        )

        dict_spectra = {
            "spectrumkykx_EK": spectrumkykx_EK,
            "spectrumkykx_EA": spectrumkykx_EA,
            "spectrumkykx_ap": spectrumkykx_ap,
            "spectrumkykx_am": spectrumkykx_am,
        }

        return dict_spectra

    # def _online_plot_saving(self, dict_spectra):
    #     pass

    def plot(self, key=None, tmin=None, tmax=None, xlim=None, zlim=None):
        """
        Plots spectrumkykx averaged between tmin and tmax.

        Parameters
        ----------
        key : str
          Key to plot the spectrum: E, EK, EA, ap_fft (default), am_fft

        tmin : float
          Lower time to compute the time average.

        tmax : float
          Upper time to compute the time average.

        xlim : float
          Upper limit kx to plot.

        zlim : float
          Upper limit kz to plot.

        """

        oper = self.sim.params.oper
        pforcing = self.sim.params.forcing

        # Load data
        with h5py.File(self.path_file, "r") as file:
            times = file["times"][...]
            kx = file["kxE"][...]
            ap_fft_spectrum = file["spectrumkykx_ap"]
            if key == "EK":
                data = file["spectrumkykx_EK"]
            elif key == "EA":
                data = file["spectrumkykx_EA"]
            elif key == "ap_fft" or not key:
                data = file["spectrumkykx_ap"]
                text_plot = r"$\hat{a}_+$"
            elif key == "am_fft":
                data = file["spectrumkykx_am"]
                text_plot = r"$\hat{a}_-$"
            else:
                raise ValueError("Key unknown.")

            # Compute time average
            if tmin is None:
                tmin = times[0]
            if tmax is None:
                tmax = times[-1]

            itmin = np.argmin(abs(times - tmin))
            if tmin == tmax:
                data_plot = data[itmin, :, :]
                if key != "ap_fft":
                    vmax_modified = ap_fft_spectrum[itmin, :, :].max()

            else:
                itmax = np.argmin(abs(times - tmax))
                data_plot = (data[itmin:itmax, :, :]).mean(0)
                if key != "ap_fft":
                    vmax_modified = (
                        (ap_fft_spectrum[itmin:itmax, :, :]).mean(0)
                    ).max()

        # Create array kz with negative values
        kz = 2 * np.pi * np.fft.fftfreq(oper.ny, oper.Ly / oper.ny)
        kz[kz.shape[0] // 2] *= -1

        # Create mesh of wave-numbers
        KX, KZ = np.meshgrid(kx, kz)

        # Set figure parameters
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$k_x$", fontsize=14)
        ax.set_ylabel(r"$k_z$", fontsize=14)

        # Set axis limit
        if xlim:
            ikx = np.argmin(abs(kx - xlim))
            ax.set_xlim([0, kx[ikx] - self.sim.oper.deltakx])
        else:
            ikx = np.argmin(abs(kx - kx.max()))

        if zlim:
            ikz = np.argmin(abs(kz - zlim))
            ikz_negative = np.argmin(abs(kz + zlim))
            ax.set_ylim([kz[ikz_negative], kz[ikz] - self.sim.oper.deltaky])
        else:
            ikz = np.argmin(abs(kz - kz.max()))

        # Modify grid
        kz_modified = np.empty_like(kz)
        kz_modified[0 : kz_modified.shape[0] // 2 - 1] = kz[
            kz_modified.shape[0] // 2 + 1 :
        ]
        kz_modified[kz_modified.shape[0] // 2 - 1 :] = kz[
            0 : kz_modified.shape[0] // 2 + 1
        ]

        KX, KZ = np.meshgrid(kx, kz_modified)

        data_plot_modified = np.empty_like(data_plot)
        data_plot_modified[0 : kz_modified.shape[0] // 2 - 1, :] = data_plot[
            kz_modified.shape[0] // 2 + 1 :, :
        ]
        data_plot_modified[kz_modified.shape[0] // 2 - 1 :, :] = data_plot[
            0 : kz_modified.shape[0] // 2 + 1, :
        ]

        # Vmax is 1% of the maximum value.
        vmin = 0
        vmax = 0.01 * data_plot_modified.max()
        if key != "ap_fft":
            vmax = 0.01 * vmax_modified

        print("vmax", vmax)

        spectrum = ax.pcolormesh(
            KX, KZ, data_plot_modified, shading="nearest", vmin=vmin, vmax=vmax
        )

        # Create a Rectangle patch
        deltak = max(self.sim.oper.deltakx, self.sim.oper.deltaky)

        if isinstance(pforcing.tcrandom_anisotropic.angle, str):
            from math import radians

            angle = radians(
                float((pforcing.tcrandom_anisotropic.angle).split("°")[0])
            )
        else:
            angle = pforcing.tcrandom_anisotropic.angle

        x_rect = np.sin(angle) * deltak * pforcing.nkmin_forcing

        z_rect = np.cos(angle) * deltak * pforcing.nkmin_forcing

        width = abs(x_rect - np.sin(angle) * deltak * pforcing.nkmax_forcing)

        height = abs(z_rect - np.cos(angle) * deltak * pforcing.nkmax_forcing)

        rect1 = patches.Rectangle(
            (x_rect, z_rect),
            width,
            height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        ax.add_patch(rect1)

        if pforcing.tcrandom_anisotropic.kz_negative_enable:
            rect2 = patches.Rectangle(
                (x_rect, -(z_rect + height)),
                width,
                height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )

            ax.add_patch(rect2)

        # Plot arc kmin and kmax forcing
        ax.add_patch(
            patches.Arc(
                xy=(0, 0),
                width=2 * pforcing.nkmin_forcing * deltak,
                height=2 * pforcing.nkmin_forcing * deltak,
                angle=0,
                theta1=-90.0,
                theta2=90.0,
                linestyle="-.",
                color="red",
            )
        )
        ax.add_patch(
            patches.Arc(
                xy=(0, 0),
                width=2 * pforcing.nkmax_forcing * deltak,
                height=2 * pforcing.nkmax_forcing * deltak,
                angle=0,
                theta1=-90,
                theta2=90.0,
                linestyle="-.",
                color="red",
            )
        )

        # Text at 70% of the xlim and ylim
        ikx_text = np.argmin(abs(kx - kx[ikx] * 0.7))
        ikz_text = np.argmin(abs(kz - kz[ikz] * 0.7))

        ax.text(
            kx[ikx_text], kz[ikz_text], f"{text_plot}", color="white", fontsize=15
        )

        # Colorbar
        fig.colorbar(spectrum)

        ax.set_aspect("equal")

        return fig
