"""Initialization of the field (:mod:`fluidsim.solvers.ns2d.init_fields`)
=========================================================================

.. autoclass:: InitFieldsNS2D
   :members:

.. autoclass:: InitFieldsNoise
   :members:

.. autoclass:: InitFieldsJet
   :members:

.. autoclass:: InitFieldsDipole
   :members:

"""

import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields


class InitFieldsNoise(SpecificInitFields):
    """Initialize the state with noise."""

    tag = "noise"

    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the `params` container (class method)."""
        super()._complete_params_with_default(params)

        p_noise = params.init_fields._set_child(
            cls.tag, attribs={"velo_max": 1.0, "length": 0.0}
        )

        p_noise._set_doc(
            """
velo_max: float (default 1.)

    Maximum velocity.

length: float (default 0.)

    The smallest (cutoff) scale in the noise.

"""
        )

    def __call__(self):
        rot_fft, ux_fft, uy_fft = self.compute_rotuxuy_fft()
        # energy_fft = 0.5 * (np.abs(ux_fft) ** 2 + np.abs(uy_fft) ** 2)
        self.sim.state.init_from_rotfft(rot_fft)

    def compute_rotuxuy_fft(self):

        params = self.sim.params
        oper = self.sim.oper

        lambda0 = params.init_fields.noise.length
        if lambda0 == 0:
            lambda0 = min(oper.Lx, oper.Ly) / 4.0

        def H_smooth(x, delta):
            return (1.0 + np.tanh(2 * np.pi * x / delta)) / 2.0

        # to compute always the same field... (for 1 resolution...)
        np.random.seed(42 + mpi.rank)

        ux_fft = (
            np.random.random(oper.shapeK)
            + 1j * np.random.random(oper.shapeK)
            - 0.5
            - 0.5j
        )
        uy_fft = (
            np.random.random(oper.shapeK)
            + 1j * np.random.random(oper.shapeK)
            - 0.5
            - 0.5j
        )

        if mpi.rank == 0:
            ux_fft[0, 0] = 0.0
            uy_fft[0, 0] = 0.0

        oper.projection_perp(ux_fft, uy_fft)
        oper.dealiasing(ux_fft, uy_fft)

        k0 = 2 * np.pi / lambda0
        delta_k0 = 1.0 * k0
        ux_fft = ux_fft * H_smooth(k0 - oper.K, delta_k0)
        uy_fft = uy_fft * H_smooth(k0 - oper.K, delta_k0)

        ux = oper.ifft2(ux_fft)
        uy = oper.ifft2(uy_fft)
        velo_max = np.sqrt(ux**2 + uy**2).max()
        if mpi.nb_proc > 1:
            velo_max = oper.comm.allreduce(velo_max, op=mpi.MPI.MAX)
        ux = params.init_fields.noise.velo_max * ux / velo_max
        uy = params.init_fields.noise.velo_max * uy / velo_max

        ux_fft = oper.fft2(ux)
        uy_fft = oper.fft2(uy)

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        return rot_fft, ux_fft, uy_fft


class InitFieldsJet(SpecificInitFields):
    """Initialize the state with a jet."""

    tag = "jet"

    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the `params` container (class method)."""
        super()._complete_params_with_default(params)

    # params.init_fields._set_child(cls.tag, attribs={})

    def __call__(self):
        oper = self.sim.oper
        rot = self.vorticity_jet()
        rot_fft = oper.fft2(rot)
        rot_fft[oper.K == 0] = 0.0
        self.sim.state.init_from_rotfft(rot_fft)

    def vorticity_jet(self):
        oper = self.sim.oper
        Ly = oper.Ly
        a = 0.5
        b = Ly / 2.0
        omega0 = 2.0
        omega = omega0 * (
            np.exp(-(((oper.YY - Ly / 2.0 + b / 2.0) / a) ** 2))
            - np.exp(-(((oper.YY - Ly / 2.0 - b / 2.0) / a) ** 2))
            + np.exp(-(((oper.YY - Ly / 2.0 + b / 2.0 + Ly) / a) ** 2))
            - np.exp(-(((oper.YY - Ly / 2.0 - b / 2.0 + Ly) / a) ** 2))
            + np.exp(-(((oper.YY - Ly / 2.0 + b / 2.0 - Ly) / a) ** 2))
            - np.exp(-(((oper.YY - Ly / 2.0 - b / 2.0 - Ly) / a) ** 2))
        )
        return omega


class InitFieldsDipole(SpecificInitFields):
    """Initialize the state with a dipole."""

    tag = "dipole"

    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the `params` container (class method)."""
        super()._complete_params_with_default(params)

    # params.init_fields._set_child(cls.tag, attribs={})

    def __call__(self):
        rot = self.vorticity_shape_1dipole()
        rot_fft = self.sim.oper.fft2(rot)
        self.sim.state.init_from_rotfft(rot_fft)

    def vorticity_shape_1dipole(self):
        oper = self.sim.oper
        xs = oper.Lx / 2.0
        ys = oper.Ly / 2.0
        theta = np.pi / 2.3
        b = 2.5
        omega = np.zeros(oper.shapeX_loc)

        def wz_2LO(XX, YY, b):
            return 2 * np.exp(-(XX**2 + (YY - b / 2.0) ** 2)) - 2 * np.exp(
                -(XX**2 + (YY + b / 2.0) ** 2)
            )

        for ip in range(-1, 2):
            for jp in range(-1, 2):
                XX_s = np.cos(theta) * (oper.XX - xs - ip * oper.Lx) + np.sin(
                    theta
                ) * (oper.YY - ys - jp * oper.Ly)
                YY_s = np.cos(theta) * (oper.YY - ys - jp * oper.Ly) - np.sin(
                    theta
                ) * (oper.XX - xs - ip * oper.Lx)
                omega = omega + wz_2LO(XX_s, YY_s, b)
        return omega


class InitFieldsNS2D(InitFieldsBase):
    """Initialize the state for the solver NS2D."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""
        InitFieldsBase._complete_info_solver(
            info_solver,
            classes=[InitFieldsNoise, InitFieldsJet, InitFieldsDipole],
        )
