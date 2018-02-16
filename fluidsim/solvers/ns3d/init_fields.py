"""Initialization of the field (:mod:`fluidsim.solvers.ns3d.init_fields`)
=========================================================================

.. autoclass:: InitFieldsNS3D
   :members:

.. autoclass:: InitFieldsDipole
   :members:

.. autoclass:: InitFieldsNoise
   :members:

"""
from __future__ import division

from builtins import range
from past.utils import old_div
import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields


class InitFieldsDipole(SpecificInitFields):
    tag = 'dipole'

    @classmethod
    def _complete_params_with_default(cls, params):
        super(InitFieldsDipole, cls)._complete_params_with_default(params)
        # params.init_fields._set_child(cls.tag, attribs={'U': 1.})

    def __call__(self):
        oper = self.sim.oper
        rot2d = self.vorticity_1dipole2d()
        rot2d_fft = oper.fft2d(rot2d)

        vx2d_fft, vy2d_fft = oper.oper2d.vecfft_from_rotfft(
            rot2d_fft)

        vx_fft = oper.build_invariant_arrayK_from_2d_indices12X(vx2d_fft)
        vy_fft = oper.build_invariant_arrayK_from_2d_indices12X(vy2d_fft)

        self.sim.state.init_from_vxvyfft(vx_fft, vy_fft)

    def vorticity_1dipole2d(self):
        oper = self.sim.oper
        xs = old_div(oper.Lx,2)
        ys = old_div(oper.Ly,2)
        theta = old_div(np.pi,2.3)
        b = 2.5
        omega = np.zeros(oper.oper2d.shapeX_loc)

        def wz_2LO(XX, YY, b):
            return (2*np.exp(-(XX**2 + (YY-old_div(b,2))**2)) -
                    2*np.exp(-(XX**2 + (YY+old_div(b,2))**2)))

        XX = oper.oper2d.XX
        YY = oper.oper2d.YY

        for ip in range(-1, 2):
            for jp in range(-1, 2):
                XX_s = (np.cos(theta) * (XX-xs-ip*oper.Lx) +
                        np.sin(theta) * (YY-ys-jp*oper.Ly))
                YY_s = (np.cos(theta) * (YY-ys-jp*oper.Ly) -
                        np.sin(theta) * (XX-xs-ip*oper.Lx))
                omega += wz_2LO(XX_s, YY_s, b)
        return omega


class InitFieldsNoise(SpecificInitFields):
    """Initialize the state with noise."""
    tag = 'noise'

    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the `params` container (class method)."""
        super(InitFieldsNoise, cls)._complete_params_with_default(params)

        params.init_fields._set_child(cls.tag, attribs={
            'velo_max': 1.,
            'length': 0})

    def __call__(self):
        vx_fft, vy_fft, vz_fft = self.compute_vv_fft()
        self.sim.state.init_from_vxvyvzfft(vx_fft, vy_fft, vz_fft)

    def compute_vv_fft(self):

        params = self.sim.params
        oper = self.sim.oper

        lambda0 = params.init_fields.noise.length
        if lambda0 == 0:
            lambda0 = oper.Lx / 4.

        def H_smooth(x, delta):
            return (1. + np.tanh(2*np.pi*x/delta))/2.

        # to compute always the same field... (for 1 resolution...)
        np.random.seed(42)  # this does not work for MPI...

        vv_fft = []
        for ii in range(3):
            vv_fft.append((np.random.random(oper.shapeK) +
                  1j*np.random.random(oper.shapeK) - 0.5 - 0.5j))
        
            if mpi.rank == 0:
                vv_fft[ii][0, 0] = 0.

        oper.project_perpk3d(*vv_fft)
        oper.dealiasing(*vv_fft)

        k0 = 2*np.pi/lambda0
        delta_k0 = 1.*k0

        KK = np.sqrt(oper.K2)
        
        vv_fft = [vi_fft*H_smooth(k0-KK, delta_k0) for vi_fft in vv_fft]
        vv = [oper.ifft(ui_fft) for ui_fft in vv_fft]

        velo_max = np.sqrt(vv[0]**2+vv[1]**2+vv[2]**2).max()
        if mpi.nb_proc > 1:
            velo_max = oper.comm.allreduce(velo_max, op=mpi.MPI.MAX)

        vv = [params.init_fields.noise.velo_max*vi/velo_max for vi in vv]

        vv_fft = [oper.fft(vi) for vi in vv]

        return tuple(vv_fft)


class InitFieldsNS3D(InitFieldsBase):
    """Init the fields for the solver NS2D."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""

        InitFieldsBase._complete_info_solver(
            info_solver, classes=[
                InitFieldsDipole, InitFieldsNoise])
