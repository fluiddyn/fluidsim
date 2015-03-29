"""Initialisation of the fields (:mod:`fluidsim.base.init_fields`)
========================================================================

.. currentmodule:: fluidsim.base.init_fields

Provides:

.. autoclass:: InitFieldsBase
   :members:
   :private-members:

"""

import numpy as np
import h5py

from copy import deepcopy

from fluiddyn.util import mpi

from fluidsim.operators.setofvariables import SetOfVariables


class InitFieldsBase(object):
    """A :class:`InitFieldsBase` object provides functions for
    initialisation of 2D fields."""

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        attribs = {'type_flow_init': 'NOISE',
                   'lambda_noise': 1.,
                   'max_velo_noise': 1.,
                   # in case type_flow_init == 'LOAD_FILE'
                   'path_file': ''}
        params.set_child('init_fields', attribs=attribs)

    implemented_flows = ['NOISE', 'CONSTANT', 'LOAD_FILE']

    def __init__(self, sim=None, oper=None, params=None):

        if sim is not None:
            self.sim = sim
            params = sim.params
            oper = sim.oper

        self.params = params
        self.oper = oper

    def get_and_check_type_flow_init(self):
        type_flow_init = self.params.init_fields.type_flow_init
        if type_flow_init not in self.implemented_flows:
            raise ValueError(type_flow_init + ' is not an implemented flows.')
        return type_flow_init

    def __call__(self):
        sim = self.sim

        type_flow_init = self.get_and_check_type_flow_init()

        if type_flow_init == 'NOISE':
            rot_fft, ux_fft, uy_fft = self.init_fields_noise()
            sim.state.state_fft['ux_fft'] = ux_fft
            sim.state.state_fft['uy_fft'] = uy_fft
            sim.state.statephys_from_statefft()

        if type_flow_init == 'LOAD_FILE':
            self.get_state_from_file(self.params.init_fields.path_file)

        elif type_flow_init == 'CONSTANT':
            sim.state.state_fft.initialize(value=1.)
            sim.state.state_phys.initialize(value=1.)

    def init_fields_1dipole(self):
        rot = self.vorticity_shape()
        rot_fft = self.oper.fft2(rot)

        self.oper.dealiasing(rot_fft)
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)

        return rot_fft, ux_fft, uy_fft

    def vorticity_shape(self):
        xs = self.oper.Lx/2
        ys = self.oper.Ly/2
        theta = np.pi/2.3
        b = 2.5
        omega = np.zeros(self.oper.shapeX_loc)

        for ip in range(-1, 2):
            for jp in range(-1, 2):
                XX_s = (np.cos(theta)*(self.oper.XX-xs-ip*self.oper.Lx)
                        + np.sin(theta)*(self.oper.YY-ys-jp*self.oper.Ly))
                YY_s = (np.cos(theta)*(self.oper.YY-ys-jp*self.oper.Ly)
                        - np.sin(theta)*(self.oper.XX-xs-ip*self.oper.Lx))
                omega = omega + self.wz_2LO(XX_s, YY_s, b)
        return omega

    def wz_2LO(self, XX, YY, b):
        return (- 2*np.exp(-(XX**2 + (YY+b/2)**2))
                + 2*np.exp(-(XX**2 + (YY-b/2)**2)))

    def init_fields_jet(self):
        rot = self.vorticity_jet()
        rot_fft = self.oper.fft2(rot)
        rot_fft[self.oper.KK == 0] = 0.
        self.oper.dealiasing(rot_fft)
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        return rot_fft, ux_fft, uy_fft

    def vorticity_jet(self):
        Ly = self.oper.Ly
        a = 0.5
        b = Ly/2
        omega0 = 2.
        # epsilon = 2.
        omega = omega0*(
            + np.exp(-((self.oper.YY - Ly/2 + b/2)/a)**2)
            - np.exp(-((self.oper.YY - Ly/2 - b/2)/a)**2)
            + np.exp(-((self.oper.YY - Ly/2 + b/2 + Ly)/a)**2)
            - np.exp(-((self.oper.YY - Ly/2 - b/2 + Ly)/a)**2)
            + np.exp(-((self.oper.YY - Ly/2 + b/2 - Ly)/a)**2)
            - np.exp(-((self.oper.YY - Ly/2 - b/2 - Ly)/a)**2)
            # + epsilon*np.random.random([self.oper.ny_loc, self.oper.nx_loc])
        )
        return omega

    def init_fields_noise(self):
        try:
            lambda0 = self.params.lambda_noise
        except AttributeError:
            lambda0 = self.oper.Lx/4
        H_smooth = lambda x, delta: (1. + np.tanh(2*np.pi*x/delta))/2

        # to compute always the same field... (for 1 resolution...)
        np.random.seed(42)  # this does not work for MPI...

        ux_fft = (np.random.random(self.oper.shapeK)
                  + 1j*np.random.random(self.oper.shapeK) - 0.5 - 0.5j)
        uy_fft = (np.random.random(self.oper.shapeK)
                  + 1j*np.random.random(self.oper.shapeK) - 0.5 - 0.5j)

        if mpi.rank == 0:
            ux_fft[0, 0] = 0.
            uy_fft[0, 0] = 0.

        self.oper.projection_perp(ux_fft, uy_fft)
        self.oper.dealiasing(ux_fft, uy_fft)

        k0 = 2*np.pi/lambda0
        delta_k0 = 1.*k0
        ux_fft = ux_fft*H_smooth(k0-self.oper.KK, delta_k0)
        uy_fft = uy_fft*H_smooth(k0-self.oper.KK, delta_k0)

        ux = self.oper.ifft2(ux_fft)
        uy = self.oper.ifft2(uy_fft)
        velo_max = np.sqrt(ux**2+uy**2).max()
        if mpi.nb_proc > 1:
            velo_max = self.oper.comm.allreduce(velo_max, op=mpi.MPI.MAX)
        ux = self.params.init_fields.max_velo_noise*ux/velo_max
        uy = self.params.init_fields.max_velo_noise*uy/velo_max
        ux_fft = self.oper.fft2(ux)
        uy_fft = self.oper.fft2(uy)

        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        return rot_fft, ux_fft, uy_fft

    def init_fields_noise_rot(self, lambda0):
        H_smooth = lambda x, delta: (1. + np.tanh(2*np.pi*x/delta))/2
        rot_fft = (np.random.random([self.nky, self.nkx])
                   + 1j*np.random.random([self.nky, self.nkx]) - 0.5 - 0.5j)
        k0 = 2*np.pi/lambda0
        delta_k0 = 1*k0
        rot_fft = rot_fft*H_smooth(k0-self.KK, delta_k0)
        self.oper.dealiasing(rot_fft)
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        ux = self.oper.ifft2(ux_fft)
        uy = self.oper.ifft2(uy_fft)
        velo_max = np.sqrt(ux**2+uy**2).max()
        if mpi.nb_proc > 1:
            velo_max = self.oper.comm.allreduce(velo_max, op=mpi.MPI.MAX)
        ux = ux/velo_max
        uy = uy/velo_max
        ux_fft = self.oper.fft2(ux)
        uy_fft = self.oper.fft2(uy)

        return rot_fft, ux_fft, uy_fft

    def init_fields_wave(self):
        ikx = self.sim.params.ikx
        eta0 = self.sim.params.eta0

        # BE CARREFUL, THIS WON'T WORK WITH MPI !!!
        if mpi.rank == 0:
            print 'init_fields_wave(ikx = {0:4d}, eta0 = {1:7.2e})'.format(
                ikx, eta0)
            print 'kx[ikx] = {0:8.2f}'.format(self.oper.kxE[ikx])

        if mpi.nb_proc > 1:
            raise ValueError('BE CARREFUL, THIS WILL BE WRONG !'
                             '  DO NOT USE THIS METHOD WITH MPI '
                             '(or rewrite it :-)')

        eta_fft = self.oper.constant_arrayK(value=0.)
        ux_fft = self.oper.constant_arrayK(value=0.)
        uy_fft = self.oper.constant_arrayK(value=0.)

        eta_fft[0, self.sim.params.ikx] = 0.1*eta0
        # eta_fft[ikx, 0] = 0.1j*eta0

        self.oper.project_fft_on_realX(eta_fft)

#        ux_fft[0,ikx] = 1.j*eta0
#        uy_fft[0,ikx] = 1.j*eta0

        div_fft = self.oper.constant_arrayK(value=0.)
        div_fft[ikx, 0] = eta0
        div_fft[0, ikx] = eta0
        self.oper.project_fft_on_realX(div_fft)
        ux_fft, uy_fft = self.oper.vecfft_from_divfft(div_fft)

        return eta_fft, ux_fft, uy_fft

    def get_state_from_file(self, path_file):
        if mpi.rank == 0:
            try:
                f = h5py.File(path_file, 'r')
            except:
                raise ValueError('file '+path_file+' is really a hd5 file?')

            print ('Load state from file:\n[...]'+path_file[-75:])

            try:
                group_oper = f['/info_simul/params/oper']
            except:
                raise ValueError(
                    'file '+path_file+' does not contain a params object')

            try:
                group_state_phys = f['/state_phys']
            except:
                raise ValueError('file ' + path_file +
                                 ' does not contain a state_phys object')

            nx_file = group_oper.attrs['nx']
            ny_file = group_oper.attrs['ny']
            Lx_file = group_oper.attrs['Lx']
            Ly_file = group_oper.attrs['Ly']

            if isinstance(nx_file, list):
                nx_file = nx_file.item()
                ny_file = ny_file.item()
                Lx_file = Lx_file.item()
                Ly_file = Ly_file.item()

            if self.params.oper.nx != nx_file:
                raise ValueError(
                    'this is not a correct state for this simulation\n'
                    'self.nx != params_file.nx')

            if self.params.oper.ny != ny_file:
                raise ValueError(
                    'this is not a correct state for this simulation\n'
                    'self.ny != params_file.ny')

            if self.params.oper.Lx != Lx_file:
                raise ValueError(
                    'this is not a correct state for this simulation\n'
                    'self.params.oper.Lx != params_file.Lx')

            if self.params.oper.Ly != Ly_file:
                raise ValueError(
                    'this is not a correct state for this simulation\n'
                    'self.params.oper.Ly != params_file.Ly')

            keys_state_phys_file = group_state_phys.keys()
        else:
            keys_state_phys_file = {}

        if mpi.nb_proc > 1:
            keys_state_phys_file = mpi.comm.bcast(keys_state_phys_file)

        state_phys = self.sim.state.state_phys
        keys_phys_needed = self.sim.info.solver.classes.State.keys_phys_needed
        for k in keys_phys_needed:
            if k in keys_state_phys_file:
                if mpi.rank == 0:
                    field_seq = group_state_phys[k][...]
                else:
                    field_seq = self.oper.constant_arrayX()

                if mpi.nb_proc > 1:
                    field_loc = self.oper.scatter_Xspace(field_seq)
                else:
                    field_loc = field_seq
                state_phys[k] = field_loc
            else:
                state_phys[k] = self.oper.constant_arrayX(value=0.)

        if mpi.rank == 0:
            t_file = group_state_phys.attrs['time']
            f.close()
        else:
            t_file = 0.

        if mpi.nb_proc > 1:
            t_file = mpi.comm.bcast(t_file)

        self.sim.state.statefft_from_statephys()
        self.sim.state.statephys_from_statefft()
        self.sim.time_stepping.t = t_file

    def get_state_from_obj_simul(self, sim_in):

        if mpi.nb_proc > 1:
            raise ValueError('BE CARREFUL, THIS WILL BE WRONG !'
                             '  DO NOT USE THIS METHOD WITH MPI')

        self.sim.time_stepping.t = sim_in.time_stepping.t

        if (self.params.oper.nx == sim_in.params.oper.nx
                and self.params.oper.ny == sim_in.params.oper.ny):
            state_fft = deepcopy(sim_in.state.state_fft)
        else:
            # modify resolution
            # state_fft = SetOfVariables('state_fft')
            state_fft = SetOfVariables(like_this_sov=self.sim.state.state_fft)
            keys_state_fft = sim_in.info.solver.classes.State['keys_state_fft']
            for k in keys_state_fft:
                field_fft_seq_in = sim_in.state.state_fft[k]
                field_fft_seq_new_res = \
                    self.sim.oper.constant_arrayK(value=0.)
                [nk0_seq, nk1_seq] = field_fft_seq_new_res.shape
                [nk0_seq_in, nk1_seq_in] = field_fft_seq_in.shape

                nk0_min = min(nk0_seq, nk0_seq_in)
                nk1_min = min(nk1_seq, nk1_seq_in)

                # it is a little bit complicate to take into account ky
                for ik1 in xrange(nk1_min):
                    field_fft_seq_new_res[0, ik1] = field_fft_seq_in[0, ik1]
                    field_fft_seq_new_res[nk0_min/2, ik1] = \
                        field_fft_seq_in[nk0_min/2, ik1]
                for ik0 in xrange(1, nk0_min/2):
                    for ik1 in xrange(nk1_min):
                        field_fft_seq_new_res[ik0, ik1] = \
                            field_fft_seq_in[ik0, ik1]
                        field_fft_seq_new_res[-ik0, ik1] = \
                            field_fft_seq_in[-ik0, ik1]

                state_fft[k] = field_fft_seq_new_res

        if self.sim.output.name_solver == sim_in.output.name_solver:
            self.sim.state.state_fft = state_fft
        else:  # complicated case... untested solution !
            # state_fft = SetOfVariables('state_fft')
            raise ValueError('Not yet implemented...')
            for k in self.sim.info.solver.classes.State['keys_state_fft']:
                if k in sim_in.info.solver.classes.State['keys_state_fft']:
                    self.sim.state.state_fft[k] = state_fft[k]
                else:
                    self.sim.state.state_fft[k] = \
                        self.oper.constant_arrayK(value=0.)

        self.sim.state.statephys_from_statefft()
