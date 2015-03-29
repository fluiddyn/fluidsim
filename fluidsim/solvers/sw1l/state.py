"""
The module :mod:`stateSW1l` supplies the class :class:`StateSW1l`.
"""

import numpy as np

from fluidsim.operators.setofvariables import SetOfVariables
from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi


class StateSW1l(StatePseudoSpectral):
    """
    The class :class:`StateSW1l` contains the variables corresponding
    to the state and handles the access to other fields for the solver
    SW1l.
    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ContainerXML info_solver.

        This is a static method!
        """
        info_solver.classes.State.set_attribs({
            'keys_state_fft': ['ux_fft', 'uy_fft', 'eta_fft'],
            'keys_state_phys': ['ux', 'uy', 'eta', 'rot'],
            'keys_computable': [],
            'keys_phys_needed': ['ux', 'uy', 'eta'],
            'keys_linear_eigenmodes': ['q_fft', 'a_fft', 'd_fft']})


    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        it = self.sim.time_stepping.it
        if (key in self.vars_computed and it == self.it_computed[key]):
            return self.vars_computed[key]

        if key == 'Jx':
            ux = self.state_phys['ux']
            eta = self.state_phys['eta']
            h = 1 + eta
            result = h*ux
        elif key == 'Jy':
            uy = self.state_phys['uy']
            eta = self.state_phys['eta']
            h = 1 + eta
            result = h*uy
        elif key == 'Jx_fft':
            Jx = self.compute('Jx')
            result = self.oper.fft2(Jx)
        elif key == 'Jy_fft':
            Jy = self.compute('Jy')
            result = self.oper.fft2(Jy)
        elif key == 'rot_fft':
            ux_fft = self.state_fft['ux_fft']
            uy_fft = self.state_fft['uy_fft']
            result = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        elif key == 'div_fft':
            ux_fft = self.state_fft['ux_fft']
            uy_fft = self.state_fft['uy_fft']
            result = self.oper.divfft_from_vecfft(ux_fft, uy_fft)
        elif key == 'div':
            div_fft = self.compute('div_fft')
            result = self.oper.ifft2(div_fft)
        elif key == 'q':
            rot = self.state_phys['rot']
            eta = self.state_phys['eta']
            result = rot-self.param.f*eta
        elif key == 'h':
            eta = self.state_phys['eta']
            result = 1 + eta

        elif key == 'Floc':
            h = self.compute('h')
            ux = self.state_phys['ux']
            uy = self.state_phys['uy']
            result = np.sqrt((ux**2 + uy**2)/(self.sim.param.c2*h))

        else:
            to_print = 'Do not know how to compute "'+key+'".'
            if RAISE_ERROR:
                raise ValueError(to_print)
            else:
                if mpi.rank == 0:
                    print(to_print
                          +'\nreturn an array of zeros.')
                    
                result = self.oper.constant_arrayX(value=0.)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result




    def statefft_from_statephys(self):
        """Compute the state in Fourier space."""
        ux = self.state_phys['ux']
        uy = self.state_phys['uy']
        eta = self.state_phys['eta']
        self.state_fft['ux_fft'] = self.oper.fft2(ux)
        self.state_fft['uy_fft'] = self.oper.fft2(uy)
        self.state_fft['eta_fft'] = self.oper.fft2(eta)

    def statephys_from_statefft(self):
        """Compute the state in physical space."""
        ifft2 = self.oper.ifft2
        ux_fft = self.state_fft['ux_fft']
        uy_fft = self.state_fft['uy_fft']
        eta_fft = self.state_fft['eta_fft']
        self.state_phys['ux'] = ifft2(ux_fft)
        self.state_phys['uy'] = ifft2(uy_fft)
        self.state_phys['eta'] = ifft2(eta_fft)
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        self.state_phys['rot'] = ifft2(rot_fft)

    def return_statephys_from_statefft(self, state_fft=None):
        """Return the state in physical space."""
        ifft2 = self.oper.ifft2
        if state_fft is None:
            state_fft = self.state_fft
        ux_fft = state_fft['ux_fft']
        uy_fft = state_fft['uy_fft']
        eta_fft = state_fft['eta_fft']
        state_phys = SetOfVariables(like_this_sov=self.state_phys)
        state_phys['ux'] = ifft2(ux_fft)
        state_phys['uy'] = ifft2(uy_fft)
        state_phys['eta'] = ifft2(eta_fft)

        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        state_phys['rot'] = ifft2(rot_fft)

        return state_phys

