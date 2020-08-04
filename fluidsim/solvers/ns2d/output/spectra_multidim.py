"""Multidimensional spectra output (:mod:`fluidsim.solvers.ns2d.output.spectra_multidim`)
=========================================================================================

.. autoclass:: SpectraMultiDimNS2D
   :members:
   :private-members:

"""

import numpy as np

from fluidsim.base.output.spectra_multidim import SpectraMultiDim


class SpectraMultiDimNS2D(SpectraMultiDim):
    """Save and plot multidimensional spectra."""

    def compute(self):
        """Compute multidimensional spectra at one time."""
        energy_fft = self.output.compute_energy_fft()

        # Computes spectrum Vs kykx
        spectrumkykx_E = self.oper.compute_spectrum_kykx(energy_fft)

        # Save results into dictionary
        dict_spectra = {"spectrumkykx_E": spectrumkykx_E}

        return dict_spectra

    def online_plot_saving(self, dict_spectra):
        raise NotImplementedError("Online plot saving is not implemented.")

    def plot(self, tmin=None, tmax=None):
        """Plots spectrumkykx averaged between tmin and tmax."""

        dict_results = self.load_mean(tmin, tmax)
        kx = dict_results["kxE"]
        ky = dict_results["kyE"]
        spectrumkykx_E = dict_results["spectrumkykx_E"]

        fig, ax = self.output.figure_axe()
        ax.set_xlabel("$k_x$")
        ax.set_ylabel("$k_y$")

        KX, KY = np.meshgrid(kx, ky)
        ax.pcolormesh(
            KX,
            KY,
            spectrumkykx_E,
            shading="nearest",
            vmin=spectrumkykx_E.min(),
            vmax=spectrumkykx_E.max(),
        )
