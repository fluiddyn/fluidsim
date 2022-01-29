"""
Plate2d InitFields (:mod:`fluidsim.solvers.plate2d.init_fields`)
======================================================================


"""

import numpy as np

from fluiddyn.util import mpi
from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields


class InitFieldsNoise(SpecificInitFields):
    tag = "noise"

    @classmethod
    def _complete_params_with_default(cls, params):
        super(cls, cls)._complete_params_with_default(params)
        params.init_fields._set_child(
            cls.tag, attribs={"velo_max": 1.0, "length": 0}
        )

    def __call__(self):
        w_fft, z_fft = self.compute_wz_fft()
        self.sim.state.init_state_from_wz_fft(w_fft, z_fft)

    def compute_wz_fft(self):
        params = self.sim.params
        oper = self.sim.oper

        lambda0 = params.init_fields.noise.length
        if lambda0 == 0:
            lambda0 = oper.Lx / 4.0

        def H_smooth(x, delta):
            return (1.0 + np.tanh(2 * np.pi * x / delta)) / 2.0

        # to compute always the same field... (for 1 resolution...)
        np.random.seed(42)  # this does not work for MPI...

        w_fft = (
            np.random.random(oper.shapeK)
            + 1j * np.random.random(oper.shapeK)
            - 0.5
            - 0.5j
        )
        z_fft = (
            np.random.random(oper.shapeK)
            + 1j * np.random.random(oper.shapeK)
            - 0.5
            - 0.5j
        )

        if mpi.rank == 0:
            w_fft[0, 0] = 0.0
            z_fft[0, 0] = 0.0

        oper.dealiasing(w_fft, z_fft)

        k0 = 2 * np.pi / lambda0
        delta_k0 = 1.0 * k0
        w_fft = w_fft * H_smooth(k0 - oper.K, delta_k0)
        z_fft = z_fft * H_smooth(k0 - oper.K, delta_k0)

        w = oper.ifft2(w_fft)
        z = oper.ifft2(z_fft)
        velo_max = np.sqrt(w**2 + z**2).max()
        if mpi.nb_proc > 1:
            velo_max = oper.comm.allreduce(velo_max, op=mpi.MPI.MAX)
        w = params.init_fields.noise.velo_max * w / velo_max
        z = params.init_fields.noise.velo_max * z / velo_max
        w_fft = oper.fft2(w)
        z_fft = oper.fft2(z)

        return w_fft, z_fft


class InitFieldsHarmonic(SpecificInitFields):
    tag = "harmonic"

    @classmethod
    def _complete_params_with_default(cls, params):
        super(cls, cls)._complete_params_with_default(params)

        params.init_fields._set_child(
            cls.tag,
            attribs={"i0": 20, "i1": 25},
        )

    def __call__(self):
        w_fft, z_fft = self.compute_wz_fft()
        self.sim.state.init_state_from_wz_fft(w_fft, z_fft)

    def compute_wz_fft(self):
        p_harmonic = self.sim.params.init_fields.harmonic
        i0 = p_harmonic.i0
        i1 = p_harmonic.i1

        oper = self.sim.oper

        w_fft = np.zeros(oper.shapeK_loc, dtype=np.complex128)
        z_fft = np.zeros(oper.shapeK_loc, dtype=np.complex128)
        w_fft[i0, i1] = 1.0
        z_fft[i0, i1] = 1.0

        w = oper.ifft2(w_fft)
        z = oper.ifft2(z_fft)

        w_fft = oper.fft2(w)
        z_fft = oper.fft2(z)

        return w_fft, z_fft


class InitFieldsPlate2D(InitFieldsBase):
    """Init the fields for the solver PLATE2D."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""

        InitFieldsBase._complete_info_solver(
            info_solver, classes=[InitFieldsNoise, InitFieldsHarmonic]
        )
