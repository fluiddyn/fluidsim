"""Initialization of the field (:mod:`fluidsim.solvers.ns3d.init_fields`)
=========================================================================

.. autoclass:: InitFieldsNS3D
   :members:

.. autoclass:: InitFieldsDipole
   :members:

.. autoclass:: InitFieldsNoise
   :members:

"""

import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields


class SpecificInitFieldsNS3D(SpecificInitFields):
    def init_state_from_fieldsfft(self, **fields):
        self.sim.state.init_statespect_from(**fields)
        self.sim.project_state_spect(self.sim.state.state_spect)
        self.sim.state.statephys_from_statespect()


class InitFieldsDipole(SpecificInitFieldsNS3D):
    tag = "dipole"

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)

    # params.init_fields._set_child(cls.tag, attribs={'U': 1.})

    def __call__(self):
        oper = self.sim.oper
        rot2d = self.vorticity_1dipole2d()
        rot2d_fft = oper.fft2d(rot2d)

        vx2d_fft, vy2d_fft = oper.oper2d.vecfft_from_rotfft(rot2d_fft)

        vx_fft = oper.build_invariant_arrayK_from_2d_indices12X(vx2d_fft)
        vy_fft = oper.build_invariant_arrayK_from_2d_indices12X(vy2d_fft)

        fields = {"vx_fft": vx_fft, "vy_fft": vy_fft}
        self.init_state_from_fieldsfft(**fields)

    def vorticity_1dipole2d(self):
        oper = self.sim.oper
        xs = oper.Lx / 2.0
        ys = oper.Ly / 2.0
        theta = np.pi / 2.3
        b = 2.5
        omega = np.zeros(oper.oper2d.shapeX_loc)

        def wz_2LO(XX, YY, b):
            return 2 * np.exp(-(XX**2 + (YY - (b / 2.0)) ** 2)) - 2 * np.exp(
                -(XX**2 + (YY + (b / 2.0)) ** 2)
            )

        XX = oper.oper2d.XX
        YY = oper.oper2d.YY

        for ip in range(-1, 2):
            for jp in range(-1, 2):
                XX_s = np.cos(theta) * (XX - xs - ip * oper.Lx) + np.sin(
                    theta
                ) * (YY - ys - jp * oper.Ly)
                YY_s = np.cos(theta) * (YY - ys - jp * oper.Ly) - np.sin(
                    theta
                ) * (XX - xs - ip * oper.Lx)
                omega += wz_2LO(XX_s, YY_s, b)
        return omega


class InitFieldsNoise(SpecificInitFieldsNS3D):
    """Initialize the state with noise."""

    tag = "noise"

    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the `params` container (class method)."""
        super()._complete_params_with_default(params)

        p_noise = params.init_fields._set_child(
            cls.tag, attribs={"velo_max": 1.0, "length": None}
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
        vx_fft, vy_fft, vz_fft = self.compute_vv_fft()
        fields = {"vx_fft": vx_fft, "vy_fft": vy_fft, "vz_fft": vz_fft}
        params = self.sim.params
        oper = self.sim.oper

        lambda0 = params.init_fields.noise.length
        if lambda0 is None:
            lambda0 = min(oper.Lx, oper.Ly, oper.Lz) / 4.0

        def H_smooth(x, delta):
            return (1.0 + np.tanh(2 * np.pi * x / delta)) / 2.0

        k0 = 2 * np.pi / lambda0
        delta_k0 = 1.0 * k0

        K = np.sqrt(oper.K2)
        velo_max = params.init_fields.noise.velo_max

        for key in self.sim.state.keys_state_spect:
            if key not in fields:
                field = np.random.random(oper.shapeX_loc)
                field_fft = oper.fft(field)
                if mpi.rank == 0:
                    field_fft[0, 0, 0] = 0.0

                field_fft *= H_smooth(k0 - K, delta_k0)
                oper.ifft_as_arg(field_fft, field)

                value_max = np.abs(field).max()
                if mpi.nb_proc > 1:
                    value_max = oper.comm.allreduce(value_max, op=mpi.MPI.MAX)

                field_max = velo_max
                if key == "b_fft" and hasattr(params, "N"):
                    field_max *= params.N

                fields[key] = (field_max / value_max) * field_fft

        self.init_state_from_fieldsfft(**fields)

    def compute_vv_fft(self):
        params_noise = self.sim.params.init_fields.noise
        return compute_solenoidal_noise_fft(
            self.sim.oper, params_noise.length, params_noise.velo_max
        )


def compute_solenoidal_noise_fft(oper, length=None, velo_max=1, seed=42):
    """Compute a divergence-free (incompressible) random field"""

    lambda0 = length
    if lambda0 is None:
        lambda0 = oper.Lx / 4.0

    def H_smooth(x, delta):
        return (1.0 + np.tanh(2 * np.pi * x / delta)) / 2.0

    # to compute always the same field... (for 1 resolution...)
    np.random.seed(seed + mpi.rank)

    vv = [np.random.random(oper.shapeX_loc) - 0.5 for i in range(3)]

    vv_fft = []
    for ii, vi in enumerate(vv):
        vv_fft.append(oper.fft(vi))

        if mpi.rank == 0:
            vv_fft[ii][0, 0, 0] = 0.0

    oper.project_perpk3d(*vv_fft)
    oper.dealiasing(*vv_fft)

    k0 = 2 * np.pi / lambda0
    delta_k0 = 1.0 * k0

    K = np.sqrt(oper.K2)

    vv_fft = [vi_fft * H_smooth(k0 - K, delta_k0) for vi_fft in vv_fft]
    vv = [oper.ifft(ui_fft) for ui_fft in vv_fft]

    velo_max_result_random = np.sqrt(vv[0] ** 2 + vv[1] ** 2 + vv[2] ** 2).max()
    if mpi.nb_proc > 1:
        velo_max_result_random = oper.comm.allreduce(
            velo_max_result_random, op=mpi.MPI.MAX
        )

    vv = [velo_max * vi / velo_max_result_random for vi in vv]

    vv_fft = [oper.fft(vi) for vi in vv]

    return tuple(vv_fft)


class InitFieldsNS3D(InitFieldsBase):
    """Init the fields for the solver NS2D."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""

        InitFieldsBase._complete_info_solver(
            info_solver, classes=[InitFieldsDipole, InitFieldsNoise]
        )

    def __call__(self):
        self.sim._init_projection()
        super().__call__()
