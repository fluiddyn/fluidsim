"""Forcing schemes (:mod:`fluidsim.solvers.ns2d.strat.forcing.base`)
====================================================================

Provides:

.. autoclass:: ForcingBasePseudoSpectralAnisotrop
   :members:
   :private-members:

"""

from fluidsim.base.forcing.base import ForcingBase


class ForcingBasePseudoSpectralAnisotrop(ForcingBase):
    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container.
        """
        ForcingBase._complete_params_with_default(params, info_solver)

        # Attributes for the anisotropic forcing
        # Anisotropic forcing: angle, nkxmax_forcing, nkxmin_forcing
        # Isotropic forcing: nkmax_forcing, nkmin_forcing
        params.forcing._set_attribs(
            {'angle': 45, 'nkxmax_forcing': 5, 'nkxmin_forcing': 4,
             'nkmax_forcing': 5, 'nkmin_forcing': 4})

    def compute(self):
        self._forcing.compute()

    def get_forcing(self):
        return self._forcing.forcing_fft
