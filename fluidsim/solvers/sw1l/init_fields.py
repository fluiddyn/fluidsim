"""Initialisation of the fields (:mod:`fluidsim.solvers.sw1l.init_fields`)
==========================================================================

Provides:

.. autoclass:: InitFieldsNoise
   :members:
   :private-members:

.. autoclass:: InitFieldsWave
   :members:
   :private-members:

.. autoclass:: InitFieldsVortexGrid
   :members:
   :private-members:

.. autoclass:: InitFieldsSW1L
   :members:
   :private-members:

"""

import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields

from fluidsim.solvers.ns2d.init_fields import (
    InitFieldsNoise as InitFieldsNoiseNS2D,
)

from fluidsim.solvers.ns2d.init_fields import InitFieldsJet, InitFieldsDipole


class InitFieldsNoise(InitFieldsNoiseNS2D):
    def __call__(self):
        rot_fft, ux_fft, uy_fft = self.compute_rotuxuy_fft()
        self.sim.state.init_from_uxuyfft(ux_fft, uy_fft)


class InitFieldsWave(SpecificInitFields):
    tag = "wave"

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)
        params.init_fields._set_child(cls.tag, attribs={"eta_max": 1.0, "ikx": 2})

    def __call__(self):
        oper = self.sim.oper

        ikx = self.sim.params.init_fields.wave.ikx
        eta_max = self.sim.params.init_fields.wave.eta_max

        kx = oper.deltakx * ikx
        eta_fft = np.zeros_like(self.sim.state.get_var("eta_fft"))
        cond = np.logical_and(oper.KX == kx, oper.KY == 0.0)
        eta_fft[cond] = eta_max
        oper.project_fft_on_realX(eta_fft)

        self.sim.state.init_from_etafft(eta_fft)


class InitFieldsVortexGrid(SpecificInitFields):
    """Initializes the vorticity field with n_vort^2 Gaussian vortices in a square
    grid.

    Vortices are randomly assigned clockwise / anti-clockwise directions;
    with equal no of vortices for each direction.

    Parameters
    ----------
    omega_max : Max vorticity of a single vortex at its center

    n_vort :

      No. of vortices along one edge of the square grid, should be even integer

    sd : Standard Deviation of the gaussian, optional
      If not specified, follows six-sigma rule based on half vortex spacing

    """

    tag = "vortex_grid"

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)
        params.init_fields._set_child(
            cls.tag, attribs={"omega_max": 1.0, "n_vort": 8, "sd": None}
        )

    def __call__(self):
        rot = self.vortex_grid_shape()
        rot_fft = self.sim.oper.fft2(rot)
        self.sim.state.init_from_rotfft(rot_fft)

    def vortex_grid_shape(self):
        oper = self.sim.oper
        params = self.sim.params.init_fields.vortex_grid

        Lx = oper.Lx
        Ly = oper.Ly
        XX = oper.XX
        YY = oper.YY
        N_vort = params.n_vort
        SD = params.sd

        if N_vort % 2 != 0:
            raise ValueError(
                "Cannot initialize a net circulation free field."
                "N_vort should be even."
            )

        dx_vort = Lx / N_vort
        dy_vort = Ly / N_vort
        x_vort = np.linspace(0, Lx, N_vort + 1) + dx_vort / 2.0
        y_vort = np.linspace(0, Ly, N_vort + 1) + dy_vort / 2.0
        sign_list = self._random_plus_minus_list()

        if SD is None:
            SD = min(dx_vort, dy_vort) / 12.0
            params.sd = SD

        amp = params.omega_max

        def wz_gaussian(x, y, sign):
            return sign * amp * np.exp(-(x**2 + y**2) / (2 * SD**2))

        omega = np.zeros(oper.shapeX_loc)
        for i in range(0, N_vort):
            x0 = x_vort[i]
            for j in range(0, N_vort):
                y0 = y_vort[j]
                sign = sign_list[i * N_vort + j]
                omega = omega + wz_gaussian(XX - x0, YY - y0, sign)

        return omega

    def _random_plus_minus_list(self):
        """
        Returns a list with of length n_vort^2, with equal number of pluses and minuses.
        """
        N = self.sim.params.init_fields.vortex_grid.n_vort**2
        if mpi.rank == 0:
            pm = np.ones(N)
            pm[::2] = -1
            np.random.shuffle(pm)
        else:
            pm = np.empty(N)

        if mpi.nb_proc > 1:
            from mpi4py.MPI import INT

            mpi.comm.Bcast([pm, INT])

        return pm


class InitFieldsSW1L(InitFieldsBase):
    """Init the fields for the solver SW1L."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""

        InitFieldsBase._complete_info_solver(
            info_solver,
            classes=[
                InitFieldsNoise,
                InitFieldsJet,
                InitFieldsDipole,
                InitFieldsWave,
                InitFieldsVortexGrid,
            ],
        )
