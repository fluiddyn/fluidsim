"""Spectra output (:mod:`fluidsim.solvers.ns2d.output.spectra`)
===============================================================

.. autoclass:: SpectraNS2D
   :members:
   :private-members:

"""

from fluidsim.base.output.spectra import Spectra


class SpectraNS2D(Spectra):
    """Save and plot spectra."""

    def compute(self):
        """compute the values at one time."""
        energy_fft = self.output.compute_energy_fft()
        # compute the spectra 1D
        spectrum1Dkx_E, spectrum1Dky_E = self.spectra1D_from_fft(energy_fft)
        dict_spectra1D = {
            "spectrum1Dkx_E": spectrum1Dkx_E,
            "spectrum1Dky_E": spectrum1Dky_E,
        }
        # compute the spectra 2D
        spectrum2D_E = self.spectrum2D_from_fft(energy_fft)
        dict_spectra2D = {"spectrum2D_E": spectrum2D_E}
        return dict_spectra1D, dict_spectra2D

    def _online_plot_saving(self, dict_spectra1D, dict_spectra2D):
        if (
            self.nx == self.params.oper.ny
            and self.params.oper.Lx == self.params.oper.Ly
        ):
            spectrum2D = dict_spectra2D["spectrum2D_E"]
            khE = self.oper.khE
            coef_norm = khE ** (3.0)
            self.ax.loglog(khE, spectrum2D * coef_norm, "k")
            lin_inf, lin_sup = self.ax.get_ylim()
            if lin_inf < 10e-6:
                lin_inf = 10e-6
            self.ax.set_ylim([lin_inf, lin_sup])
        else:
            print(
                "you need to implement the ploting of the spectra for this case"
            )

    def plot1d(
        self,
        tmin=0,
        tmax=1000,
        delta_t=None,
        with_average=True,
        coef_compensate=3,
        coef_plot_k3=None,
        coef_plot_k53=None,
        coef_plot_k2=None,
        xlim=None,
        ylim=None,
        directions=None,
    ):
        self._plot_ndim(
            tmin=tmin,
            tmax=tmax,
            delta_t=delta_t,
            with_average=with_average,
            coef_compensate=coef_compensate,
            coef_plot_k3=coef_plot_k3,
            coef_plot_k53=coef_plot_k53,
            coef_plot_k2=coef_plot_k2,
            xlim=xlim,
            ylim=ylim,
            ndim=1,
            directions=directions,
        )

    def plot2d(
        self,
        tmin=0,
        tmax=1000,
        delta_t=None,
        with_average=True,
        coef_compensate=3,
        coef_plot_k3=None,
        coef_plot_k53=None,
        coef_plot_k2=None,
        xlim=None,
        ylim=None,
        directions=None,
    ):
        self._plot_ndim(
            tmin=tmin,
            tmax=tmax,
            delta_t=delta_t,
            with_average=with_average,
            coef_compensate=coef_compensate,
            coef_plot_k3=coef_plot_k3,
            coef_plot_k53=coef_plot_k53,
            coef_plot_k2=coef_plot_k2,
            xlim=xlim,
            ylim=ylim,
            ndim=2,
            directions=directions,
        )
