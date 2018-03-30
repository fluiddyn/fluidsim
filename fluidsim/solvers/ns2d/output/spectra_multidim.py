"""Multidimensional spectra output (:mod:`fluidsim.solvers.ns2d.output.spectra_multidim`)
=========================================================================================

.. autoclass:: SpectraMultiDimNS2D
   :members:
   :private-members:

"""
from __future__ import division
from __future__ import print_function

from fluidsim.base.output.spectra_multidim import SpectraMultiDim


class SpectraMultiDimNS2D(SpectraMultiDim):
    """Save and plot multidimensional spectra."""

    def compute(self):
        """Compute multidimensional spectra at one time."""
        energy_fft = self.output.compute_energy_fft()
        
        # Computes spectrum Vs kykx
        spectrumkykx_E = self.oper.compute_spectrum_kykx(energy_fft)

        # Save results into dictionary
        dico_spectra = {'spectrumkykx_E' : spectrumkykx_E}

        return dico_spectra

    def online_plot_saving(self, dico_spectra):
        raise NotImplementedError('Online plot saving is not implemented.')

    def plot(self):
        """Plots spectrumkykx averaged between tmin and tmax."""
        
        dico_results = self.load_mean(tmin, tmax)
        kx = dico_results['kxE']
        ky = dico_results['kyE']
        spectrumkykx_E = dico_results['spectrumkykx_E']
        
        fig, ax = self.output.figure_axe()
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        
        KX, KY = np.meshgrid(kx, ky)
        ax.pcolormesh(KX, KY, spectrumkykx_E,
                      vmin=spectrumkykx_E.min(), vmax=spectrumkykx_E.max())
        
