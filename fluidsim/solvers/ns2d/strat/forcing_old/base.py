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

        Isotropic forcing
        =================
        Forcing all fourier modes within a shell of thickness
        (nkmax_forcing-nkmin_forcing)*deltakx.
        Where deltakx = 2*pi/L_x.

        nkmax_forcing : Upper limit of forcing (int).

        nkmin_forcing : Lower limit of forcing (int).

        Anisotropic forcing
        ===================
        Forcing fourier modes in a rectangular vertical band.

        angle : angle that sets the kz_max (degrees).

        nkxmax_forcing : Upper limit of kx (int).

        nkxmin_forcing : Lower limit of kx (int).
        """
        ForcingBase._complete_params_with_default(params, info_solver)

        params.forcing._set_attribs(
            {'angle': 45, 'nkxmax_forcing': 5, 'nkxmin_forcing': 4,
             'nkmax_forcing': 5, 'nkmin_forcing': 4})

    def compute(self):
        self._forcing.compute()

    def get_forcing(self):
        return self._forcing.forcing_fft
