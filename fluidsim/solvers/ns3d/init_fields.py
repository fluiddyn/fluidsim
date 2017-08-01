"""Initialization of the field (:mod:`fluidsim.solvers.ns3d.init_fields`)
=========================================================================

.. autoclass:: InitFieldsNS3D
   :members:

.. autoclass:: InitFieldsDipole
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


class InitFieldsNS3D(InitFieldsBase):
    """Init the fields for the solver NS2D."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver."""

        InitFieldsBase._complete_info_solver(
            info_solver, classes=[
                InitFieldsDipole])
