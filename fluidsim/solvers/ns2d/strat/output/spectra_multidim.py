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
        energy_fft = energyK_fft + energyA_fft

        ap_fft = self.sim.state.compute("ap_fft")
        am_fft = self.sim.state.compute("am_fft")

        # Computes multidimensional spectra
        spectrumkykx_E = self.oper.compute_spectrum_kykx(energy_fft, folded=False)
        spectrumkykx_EK = self.oper.compute_spectrum_kykx(energyK_fft, folded=False)
        spectrumkykx_EA = self.oper.compute_spectrum_kykx(energyA_fft, folded=False)

        # The function compute_spectrum_kykx does not supports complex variable...
        # Only works for the energy!

        energy_ap_fft = abs(ap_fft)**2
        energy_am_fft = abs(am_fft)**2

        spectrumkykx_ap_fft = self.oper.compute_spectrum_kykx(energy_ap_fft, folded=False)
        spectrumkykx_am_fft = self.oper.compute_spectrum_kykx(energy_am_fft, folded=False)

        dict_spectra = {
            "spectrumkykx_E": spectrumkykx_E,
            "spectrumkykx_EK": spectrumkykx_EK,
            "spectrumkykx_EA": spectrumkykx_EA,
            "spectrumkykx_ap_fft": spectrumkykx_ap_fft,
            "spectrumkykx_am_fft": spectrumkykx_am_fft
        }

        return dict_spectra

    def _online_plot_saving(self, dict_spectra):
        raise NotImplementedError("_online_plot_saving in not implemented.")

    def plot(self, key=None, tmin=0, tmax=None):
        """
        Plots spectrumkykx averaged between tmin and tmax.

        Parameters
        ----------
        key : str
          Key to plot the spectrum: E, EK, EA, ap_fft (default), am_fft

        """

        oper = self.sim.params.oper

        # Load data
        with h5py.File(self.path_file, "r") as f:
            times = f["times"].value
            kx = f["kxE"].value
            # kz = f["kyE"].value
            if key == "E":
                data = f["spectrumkykx_E"].value
            elif key == "EK":
                data = f["spectrumkykx_EK"].value
            elif key == "EA":
                data = f["spectrumkykx_EA"].value
            elif key == "ap_fft" or not key:
                data = f["spectrumkykx_ap_fft"].value
            elif key == "am_fft":
                data = f["spectrumkykx_am_fft"].value
            else:
                raise ValueError("Key unknown.")

        # Compute time average
        if not tmax:
            tmax = times[-1]

        itmin = np.argmin(abs(times - tmin))
        itmax = np.argmin(abs(times - tmax))

        data_plot = np.mean(data[itmin : itmax, :, :], axis=0)

        # Create array kz with negative values
        kz = 2 * np.pi * np.fft.fftfreq(oper.ny, oper.Ly / oper.ny)
        kz[kz.shape[0]//2] *= -1

        # Create mesh of wave-numbers
        KX, KZ = np.meshgrid(kx, kz)

        ### Data
        ikx = np.argmin(abs(kx - 200))
        ikz = np.argmin(abs(kz - 148))
        ikz_negative = np.argmin(abs(kz + 148))


        # Set figure parameters
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_z$")

        kz_modified = np.empty_like(kz)
        kz_modified[0:kz_modified.shape[0]//2 - 1] = kz[kz_modified.shape[0]//2 + 1:]
        kz_modified[kz_modified.shape[0]//2 - 1:] = kz[0:kz_modified.shape[0]//2 + 1]

        KX, KZ = np.meshgrid(kx, kz_modified)

        data_plot_modified = np.empty_like(data_plot)
        data_plot_modified[0:kz_modified.shape[0]//2 - 1, :] = data_plot[kz_modified.shape[0]//2 + 1:, :]
        data_plot_modified[kz_modified.shape[0]//2 - 1:, :] = data_plot[0:kz_modified.shape[0]//2 + 1, :]

        ax.pcolormesh(KX, KZ, data_plot_modified)


        # Create a Rectangle patch
        deltak = max(self.sim.oper.deltakx, self.sim.oper.deltaky)

        angle = radians(float(self.sim.params.forcing.tcrandom_anisotropic.angle.split("°")[0]))

        x_rect = np.sin(angle) * deltak * self.sim.params.forcing.nkmin_forcing

        z_rect = np.cos(angle) * deltak * self.sim.params.forcing.nkmin_forcing

        width = abs(x_rect - np.sin(angle) * deltak * self.sim.params.forcing.nkmax_forcing)

        height = abs(z_rect - np.cos(angle) * deltak * self.sim.params.forcing.nkmax_forcing)

        rect1 = patches.Rectangle((x_rect,z_rect),width,height,linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect1)

        if self.sim.params.forcing.tcrandom_anisotropic.kz_negative_enable:
            rect2 = patches.Rectangle(
                (x_rect,-(z_rect + height)), width, height, linewidth=1,
                edgecolor='r',facecolor='none')

            ax.add_patch(rect2)

        # Plot arc kmin and kmax forcing
        ax.add_patch(
            patches.Arc(
                xy=(0, 0),
                width=2 * self.sim.params.forcing.nkmin_forcing * deltak,
                height=2 * self.sim.params.forcing.nkmin_forcing * deltak,
                angle=0,
                theta1=-90.,
                theta2=90.,
                linestyle="-.",
                color="red"
            )
        )
        ax.add_patch(
            patches.Arc(
                xy=(0, 0),
                width=2 * self.sim.params.forcing.nkmax_forcing * deltak,
                height=2 * self.sim.params.forcing.nkmax_forcing * deltak,
                angle=0,
                theta1=-90,
                theta2=90.,
                linestyle="-.",
                color="red"
            )
        )

        ax.set_aspect("equal")
