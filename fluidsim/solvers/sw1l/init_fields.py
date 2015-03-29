
"""InitFieldsSW1l"""

import os

from fluiddyn.util import mpi

from fluidsim.base.init_fields import InitFieldsBase


class InitFieldsSW1l(InitFieldsBase):
    """Init the fields for the solver SW1l."""

    implemented_flows = ['NOISE', 'CONSTANT', 'LOAD_FILE',
                         'DIPOLE', 'JET', 'WAVE']

    def __call__(self):
        """Initialization initial fields"""
        sim = self.sim

        sim.time_stepping.t = 0.
        sim.time_stepping.it = 0
        oper = sim.oper

        type_flow_init = self.get_and_check_type_flow_init()

        if type_flow_init == 'DIPOLE':
            rot_fft, ux_fft, uy_fft = self.init_fields_1dipole()
            self.fill_state_from_uxuyfft(ux_fft, uy_fft)
        elif type_flow_init == 'JET':
            rot_fft, ux_fft, uy_fft = self.init_fields_jet()
            self.fill_state_from_uxuyfft(ux_fft, uy_fft)
        elif type_flow_init == 'NOISE':
            rot_fft, ux_fft, uy_fft = self.init_fields_noise()
            self.fill_state_from_uxuyfft(ux_fft, uy_fft)
        elif type_flow_init == 'LOAD_FILE':
            path_file = sim.params.init_fields.path_file
            if not os.path.exists(path_file):
                raise ValueError('file \"{0}\" not found'.format(path_file))
            self.get_state_from_file(path_file)
        elif type_flow_init == 'CONSTANT':
            ux_fft = oper.constant_arrayK(value=1.)
            uy_fft = oper.constant_arrayK(value=0.)
            self.fill_state_from_uxuyfft(ux_fft, uy_fft)
        elif type_flow_init == 'WAVE':
            eta_fft, ux_fft, uy_fft = \
                self.init_fields_wave()
            self.fill_state_from_uxuyetafft(ux_fft, uy_fft, eta_fft)
        else:
            raise ValueError('bad value of params.init_fields.type_flow_init')

    def fill_state_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft):
        sim = self.sim
        state_fft = sim.state.state_fft
        state_fft['ux_fft'] = ux_fft
        state_fft['uy_fft'] = uy_fft
        state_fft['eta_fft'] = eta_fft

        sim.oper.dealiasing(state_fft)
        sim.state.statephys_from_statefft()

    def fill_state_from_uxuyfft(self, ux_fft, uy_fft):
        sim = self.sim
        oper = sim.oper
        ifft2 = oper.ifft2

        oper.projection_perp(ux_fft, uy_fft)
        oper.dealiasing(ux_fft, uy_fft)

        ux = ifft2(ux_fft)
        uy = ifft2(uy_fft)

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        rot = ifft2(rot_fft)

        eta_fft = self.etafft_no_div(ux, uy, rot)
        eta = ifft2(eta_fft)

        state_fft = sim.state.state_fft
        state_fft['ux_fft'] = ux_fft
        state_fft['uy_fft'] = uy_fft
        state_fft['eta_fft'] = eta_fft

        state_phys = sim.state.state_phys
        state_phys['rot'] = rot
        state_phys['ux'] = ux
        state_phys['uy'] = uy
        state_phys['eta'] = eta

    def etafft_no_div(self, ux, uy, rot):
        K2_not0 = self.oper.K2_not0
        rot_abs = rot + self.params.f

        tempx_fft = -self.oper.fft2(rot_abs*uy)
        tempy_fft = +self.oper.fft2(rot_abs*ux)

        uu2_fft = self.oper.fft2(ux**2+uy**2)

        eta_fft = (1.j * self.oper.KX*tempx_fft/K2_not0 +
                   1.j*self.oper.KY*tempy_fft/K2_not0 -
                   uu2_fft/2)/self.params.c2
        if mpi.rank == 0:
            eta_fft[0, 0] = 0.
        self.oper.dealiasing(eta_fft)

        return eta_fft


class InitFieldsSW1lExLin(InitFieldsSW1l):
    """Init the fields for the solver SW1lExLin."""

    def fill_state_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft):
        sim = self.sim

        (q_fft, ap_fft, am_fft
         ) = self.oper.qapamfft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        # q_fft = self.oper.constant_arrayK(value=0)
        # ap_fft = self.oper.constant_arrayK(value=0)
        # am_fft = self.oper.constant_arrayK(value=0)

        # am_fft[0,8] = 1.

        state_fft = sim.state.state_fft
        state_fft['q_fft'] = q_fft
        state_fft['ap_fft'] = ap_fft
        state_fft['am_fft'] = am_fft

        sim.oper.dealiasing(state_fft)
        sim.state.statephys_from_statefft()

    def fill_state_from_uxuyfft(self, ux_fft, uy_fft):
        sim = self.sim
        oper = sim.oper
        ifft2 = oper.ifft2

        oper.projection_perp(ux_fft, uy_fft)
        oper.dealiasing(ux_fft, uy_fft)

        ux = ifft2(ux_fft)
        uy = ifft2(uy_fft)

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        rot = ifft2(rot_fft)

        eta_fft = self.etafft_no_div(ux, uy, rot)
        eta = ifft2(eta_fft)

        (q_fft, ap_fft, am_fft
         ) = self.oper.qapamfft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        state_fft = sim.state.state_fft
        state_fft['q_fft'] = q_fft
        state_fft['ap_fft'] = ap_fft
        state_fft['am_fft'] = am_fft

        state_phys = sim.state.state_phys
        state_phys['rot'] = rot
        state_phys['ux'] = ux
        state_phys['uy'] = uy
        state_phys['eta'] = eta


class InitFieldsSW1lWaves(InitFieldsSW1l):
    """Init """

    def fill_state_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft):
        sim = self.sim

        (q_fft, ap_fft, am_fft
         ) = self.oper.qapamfft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        # q_fft = self.oper.constant_arrayK(value=0)
        # ap_fft = self.oper.constant_arrayK(value=0)
        # am_fft = self.oper.constant_arrayK(value=0)

        # am_fft[0,8] = 1.

        state_fft = sim.state.state_fft
        state_fft['ap_fft'] = ap_fft
        state_fft['am_fft'] = am_fft

        sim.oper.dealiasing(state_fft)
        sim.state.statephys_from_statefft()

    def fill_state_from_uxuyfft(self, ux_fft, uy_fft):
        sim = self.sim
        oper = sim.oper
        ifft2 = oper.ifft2

        oper.projection_perp(ux_fft, uy_fft)
        oper.dealiasing(ux_fft, uy_fft)

        ux = ifft2(ux_fft)
        uy = ifft2(uy_fft)

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        rot = ifft2(rot_fft)

        eta_fft = self.etafft_no_div(ux, uy, rot)
        eta = ifft2(eta_fft)

        (q_fft, ap_fft, am_fft
         ) = self.oper.qapamfft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        state_fft = sim.state.state_fft
        state_fft['ap_fft'] = ap_fft
        state_fft['am_fft'] = am_fft

        state_phys = sim.state.state_phys
        state_phys['ux'] = ux
        state_phys['uy'] = uy
        state_phys['eta'] = eta
