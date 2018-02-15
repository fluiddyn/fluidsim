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
            'keys_state_spect': [k + '_fft' for k in keys_state_phys],
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
        elif key == 'ux':
            result = self.state_phys.get_var('vx')
        elif key == 'uy':
            result = self.state_phys.get_var('vy')
        elif key == 'rotz':
            vx_fft = self('vx_fft')
            vy_fft = self('vy_fft')
            rotz_fft = self.oper.rotzfft_from_vxvyfft(vx_fft, vy_fft)
            result = self.oper.ifft3d(rotz_fft)
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

    def init_from_vxvyfft(self, vx_fft, vy_fft):
        self.state_spect.set_var('vx_fft', vx_fft)
        self.state_spect.set_var('vy_fft', vy_fft)
        self.state_spect.set_var('vz_fft', np.zeros_like(vx_fft))

        self.statephys_from_statespect()
        self.statespect_from_statephys()

    def init_from_vxvyvzfft(self, vx_fft, vy_fft, vz_fft):
        self.state_spect.set_var('vx_fft', vx_fft)
        self.state_spect.set_var('vy_fft', vy_fft)
        self.state_spect.set_var('vz_fft', vz_fft)

        self.statephys_from_statespect()
        self.statespect_from_statephys()
