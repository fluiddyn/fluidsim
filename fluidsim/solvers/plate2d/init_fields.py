"""
Plate2d InitFields (:mod:`fluidsim.solvers.plate2d.init_fields`)
======================================================================


"""

import numpy as np

from fluiddyn.util import mpi
from fluidsim.base.init_fields import InitFieldsBase


class InitFieldsPlate2D(InitFieldsBase):
    """Init the fields for the solver PLATE2D."""

    implemented_flows = ['NOISE', 'CONSTANT', 'LOAD_FILE', 'HARMONIC']

    def __call__(self):
        """Init the state (in physical and Fourier space) and time"""
        sim = self.sim

        type_flow_init = self.get_and_check_type_flow_init()

        if type_flow_init == 'HARMONIC':
            w_fft, z_fft = self.init_fields_harmonic()
            tasks_complete_init = ['Fourier_to_phys']
        elif type_flow_init == 'NOISE':
            w_fft, z_fft = self.init_fields_noise()
            tasks_complete_init = ['Fourier_to_phys']
        elif type_flow_init == 'LOAD_FILE':
            self.get_state_from_file(self.params.init_fields.path_file)
            tasks_complete_init = []
        elif type_flow_init == 'CONSTANT':
            #   rot_fft = sim.oper.constant_arrayK(value=0.)
            #    if mpi.rank == 0:
            #       rot_fft[1, 0] = 1.
            tasks_complete_init = ['Fourier_to_phys']
        else:
            raise ValueError('bad value of params.type_flow_init')

        if 'Fourier_to_phys' in tasks_complete_init:
            sim.oper.dealiasing(w_fft)
            sim.oper.dealiasing(z_fft)
            sim.state.state_fft['w_fft'] = w_fft
            sim.state.state_fft['z_fft'] = z_fft

            sim.state.statephys_from_statefft()

    def init_fields_harmonic(self):
        w_fft = np.zeros(self.sim.oper.shapeK_loc, dtype=np.complex128)
        z_fft = np.zeros(self.sim.oper.shapeK_loc, dtype=np.complex128)
        w_fft[20, 25] = 1.
        z_fft[20, 25] = 1.

        w = self.oper.ifft2(w_fft)
        z = self.oper.ifft2(z_fft)

        w_fft = self.oper.fft2(w)
        z_fft = self.oper.fft2(z)

        return w_fft, z_fft

    def init_fields_noise(self):
        try:
            lambda0 = self.params.init_fields.lambda_noise
        except AttributeError:
            lambda0 = self.oper.Lx/4

        def H_smooth(x, delta):
            return (1. + np.tanh(2*np.pi*x/delta))/2

        # to compute always the same field... (for 1 resolution...)
        np.random.seed(42)  # this does not work for MPI...

        w_fft = (np.random.random(self.oper.shapeK) +
                 1j*np.random.random(self.oper.shapeK) - 0.5 - 0.5j)
        z_fft = (np.random.random(self.oper.shapeK) +
                 1j*np.random.random(self.oper.shapeK) - 0.5 - 0.5j)

        if mpi.rank == 0:
            w_fft[0, 0] = 0.
            z_fft[0, 0] = 0.

        self.oper.dealiasing(w_fft, z_fft)

        k0 = 2*np.pi/lambda0
        delta_k0 = 1.*k0
        w_fft = w_fft*H_smooth(k0-self.oper.KK, delta_k0)
        z_fft = z_fft*H_smooth(k0-self.oper.KK, delta_k0)

        w = self.oper.ifft2(w_fft)
        z = self.oper.ifft2(z_fft)
        velo_max = np.sqrt(w**2+z**2).max()
        if mpi.nb_proc > 1:
            velo_max = self.oper.comm.allreduce(velo_max, op=mpi.MPI.MAX)
        w = self.params.init_fields.max_velo_noise*w/velo_max
        z = self.params.init_fields.max_velo_noise*z/velo_max
        w_fft = self.oper.fft2(w)
        z_fft = self.oper.fft2(z)

        return w_fft, z_fft
