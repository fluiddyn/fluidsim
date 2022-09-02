"""Spectral energy budget
=========================

Provides:

.. autoclass:: SpectralEnergyBudgetBase
   :members:
   :private-members:
   :noindex:
   :undoc-members:

"""

import numpy as np

from fluiddyn.util import mpi

from .base import SpecificOutput


def cumsum_inv(a):
    return a[::-1].cumsum()[::-1]


def inner_prod(a_fft, b_fft):
    return np.real(a_fft.conj() * b_fft)


class SpectralEnergyBudgetBase(SpecificOutput):
    """Handle the saving and plotting of spectral energy budget.

    This class uses the particular functions defined by some solvers
    :func:`` and
    :func``. If the solver doesn't has these
    functions, this class does nothing.
    """

    _tag = "spect_energy_budg"
    _name_file = _tag + ".h5"

    @staticmethod
    def _complete_params_with_default(params):
        tag = "spect_energy_budg"

        params.output.periods_save._set_attrib(tag, 0)
        params.output._set_child(tag, attribs={"HAS_TO_PLOT_SAVED": False})

    def __init__(self, output):

        params = output.sim.params
        self.nx = params.oper.nx

        self.spectrum2D_from_fft = output.sim.oper.spectrum2D_from_fft
        self.spectra1D_from_fft = output.sim.oper.spectra1D_from_fft
        self.sum_wavenumbers = output.sim.oper.sum_wavenumbers

        HAS_TO_PLOT_SAVED = params.output.spect_energy_budg.HAS_TO_PLOT_SAVED
        super().__init__(
            output,
            period_save=params.output.periods_save.spect_energy_budg,
            has_to_plot_saved=HAS_TO_PLOT_SAVED,
            arrays_1st_time={
                "khE": output.sim.oper.khE,
                "kxE": output.sim.oper.kxE,
                "kyE": output.sim.oper.kyE,
            },
        )

    def compute(self):
        """compute the values at one time."""
        if mpi.rank == 0:
            dict_results = {}
            return dict_results

    def _init_online_plot(self):
        if mpi.rank == 0:
            width_axe = 0.85
            height_axe = 0.37
            x_left_axe = 0.12
            z_bottom_axe = 0.56

            size_axe = [x_left_axe, z_bottom_axe, width_axe, height_axe]
            self.fig, axe_a = self.output.figure_axe(
                size_axe=size_axe, numfig=4_000_000
            )
            self.axe_a = axe_a
            axe_a.set_xlabel(r"$k_h$")
            axe_a.set_ylabel(r"$\Pi(k_h)$")
            axe_a.set_title("energy flux\n" + self.output.summary_simul)
            axe_a.set_xscale("log")

            z_bottom_axe = 0.08
            size_axe[1] = z_bottom_axe
            axe_b = self.fig.add_axes(size_axe)
            self.axe_b = axe_b
            axe_b.set_xlabel(r"$k_h$")
            axe_b.set_ylabel(r"$\Pi(k_h)$")
            axe_b.set_xscale("log")

    def fnonlinfft_from_uxuy_funcfft(self, ux, uy, f_fft):
        r"""
        Compute a non-linear term.

        Notes
        -----
        Returns an fft-sized nd-array equivalent to the expression:

        .. math:: - \widehat{(\vec{u}.\nabla)f}
        """

        oper = self.oper
        px_f_fft, py_f_fft = oper.gradfft_from_fft(f_fft)
        px_f = oper.ifft2(px_f_fft)
        py_f = oper.ifft2(py_f_fft)
        del (px_f_fft, py_f_fft)
        Fnl = -ux * px_f - uy * py_f
        del (px_f, py_f)
        Fnl_fft = oper.fft2(Fnl)
        oper.dealiasing(Fnl_fft)
        return Fnl_fft

    def fnonlinfft_from_uruddivfunc(
        self, urx, ury, udx, udy, div, func_fft, func
    ):
        """Compute a non-linear term."""
        oper = self.oper
        px_func_fft, py_func_fft = oper.gradfft_from_fft(func_fft)
        px_func = oper.ifft2(px_func_fft)
        py_func = oper.ifft2(py_func_fft)
        del (px_func_fft, py_func_fft)
        Frf = -urx * px_func - ury * py_func
        Fdf = -udx * px_func - udy * py_func - div * func / 2
        del (px_func, py_func)
        Frf_fft = oper.fft2(Frf)
        Fdf_fft = oper.fft2(Fdf)
        oper.dealiasing(Frf_fft, Fdf_fft)
        return Frf_fft, Fdf_fft
