
"""InitFieldsNS3D"""

import numpy as np

from fluidsim.base.init_fields import InitFieldsBase, SpecificInitFields


class InitFieldsDipole(SpecificInitFields):
    tag = 'dipole'

    @classmethod
    def _complete_params_with_default(cls, params):
        super(InitFieldsDipole, cls)._complete_params_with_default(params)
        # params.init_fields._set_child(cls.tag, attribs={'U': 1.})

    def __call__(self):
        rot2d = self.vorticity_1dipole2d()
        rot2d_fft = self.sim.oper.fft2d(rot2d)

        vx2d_fft, vy2d_fft = self.sim.oper._oper2d.vecfft_from_rotfft(
            rot2d_fft)

        vx_fft = self.sim.oper.expand_3dfrom2d(vx2d_fft)
        vy_fft = self.sim.oper.expand_3dfrom2d(vy2d_fft)

        self.sim.state.init_from_vxvyfft(vx_fft, vy_fft)

    def vorticity_1dipole2d(self):
        oper = self.sim.oper
        xs = oper.Lx/2
        ys = oper.Ly/2
        theta = np.pi/2.3
        b = 2.5
        omega = np.zeros(oper._oper2d.shapeX_loc)

        def wz_2LO(XX, YY, b):
            return (2*np.exp(-(XX**2 + (YY-b/2)**2)) -
                    2*np.exp(-(XX**2 + (YY+b/2)**2)))

        XX = oper._oper2d.XX
        YY = oper._oper2d.YY

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
