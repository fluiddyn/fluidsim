"""State for the NS3D solver (:mod:`fluidsim.solvers.ns3d.state`)
=======================================================================
"""

import numpy as np


from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi


class StateNS3D(StatePseudoSpectral):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the solver NS3D.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        keys_state_phys = ['vx', 'vy', 'vz']

        info_solver.classes.State._set_attribs({
            'keys_state_fft': [k + '_fft' for k in keys_state_phys],
            'keys_state_phys': keys_state_phys,
            'keys_phys_needed': keys_state_phys,
            'keys_computable': [],
            'keys_linear_eigenmodes': ['rot_fft']
        })

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        it = self.sim.time_stepping.it
        if (key in self.vars_computed and it == self.it_computed[key]):
            return self.vars_computed[key]

        if key == 'vx_fft':
            result = self.oper.fft3d(self.state_phys.get_var('vx'))
        elif key == 'vy_fft':
            result = self.oper.fft3d(self.state_phys.get_var('vy'))
        elif key == 'vz_fft':
            result = self.oper.fft3d(self.state_phys.get_var('vz'))
        else:
            to_print = 'Do not know how to compute "' + key + '".'
            if RAISE_ERROR:
                raise ValueError(to_print)
            else:
                mpi.printby0(to_print +
                             '\nreturn an array of zeros.')

                result = self.oper.constant_arrayX(value=0.)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result

    def statephys_from_statefft(self):
        vx_fft = self.state_fft.get_var('vx_fft')
        vy_fft = self.state_fft.get_var('vy_fft')
        vz_fft = self.state_fft.get_var('vz_fft')

        self.state_phys.set_var('vx', self.oper.ifft3d(vx_fft))
        self.state_phys.set_var('vy', self.oper.ifft3d(vy_fft))
        self.state_phys.set_var('vz', self.oper.ifft3d(vz_fft))

    def statefft_from_statephys(self):
        vx = self.state_phys.get_var('vx')
        vy = self.state_phys.get_var('vy')
        vz = self.state_phys.get_var('vz')

        self.state_fft.set_var('vx_fft', self.oper.fft3d(vx))
        self.state_fft.set_var('vy_fft', self.oper.fft3d(vy))
        self.state_fft.set_var('vz_fft', self.oper.fft3d(vz))

    def init_from_vxvyfft(self, vx_fft, vy_fft):
        self.state_fft.set_var('vx_fft', vx_fft)
        self.state_fft.set_var('vy_fft', vy_fft)
        self.state_fft.set_var('vz_fft', np.zeros_like(vx_fft))

        self.state_phys.set_var('vx', self.oper.ifft3d(vx_fft))
        self.state_phys.set_var('vy', self.oper.ifft3d(vy_fft))
        vx = self.state_phys.get_var('vx')
        self.state_phys.set_var('vz', np.zeros_like(vx))
