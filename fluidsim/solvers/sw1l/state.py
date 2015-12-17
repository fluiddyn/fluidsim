"""
The module :mod:`stateSW1L` supplies the class :class:`StateSW1L`.
"""

import numpy as np

from fluidsim.base.setofvariables import SetOfVariables
from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi


class StateSW1L(StatePseudoSpectral):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the solver SW1L.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs({
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
            ux = self.state_phys.get_var('ux')
            eta = self.state_phys.get_var('eta')
            h = 1 + eta
            result = h*ux
        elif key == 'Jy':
            uy = self.state_phys.get_var('uy')
            eta = self.state_phys.get_var('eta')
            h = 1 + eta
            result = h*uy
        elif key == 'Jx_fft':
            Jx = self.compute('Jx')
            result = self.oper.fft2(Jx)
        elif key == 'Jy_fft':
            Jy = self.compute('Jy')
            result = self.oper.fft2(Jy)
        elif key == 'rot_fft':
            ux_fft = self.state_fft.get_var('ux_fft')
            uy_fft = self.state_fft.get_var('uy_fft')
            result = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        elif key == 'div_fft':
            ux_fft = self.state_fft.get_var('ux_fft')
            uy_fft = self.state_fft.get_var('uy_fft')
            result = self.oper.divfft_from_vecfft(ux_fft, uy_fft)
        elif key == 'div':
            div_fft = self.compute('div_fft')
            result = self.oper.ifft2(div_fft)
        elif key == 'q':
            rot = self.state_phys.get_var('rot')
            eta = self.state_phys.get_var('eta')
            result = rot-self.params.f*eta
        elif key == 'q_fft':
            ux_fft = self.state_fft.get_var('ux_fft')
            uy_fft = self.state_fft.get_var('uy_fft')
            eta_fft = self.state_fft.get_var('eta_fft')
            rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
            result = rot_fft-self.params.f*eta_fft

        elif key == 'a_fft':
            ux_fft = self.state_fft.get_var('ux_fft')
            uy_fft = self.state_fft.get_var('uy_fft')
            eta_fft = self.state_fft.get_var('eta_fft')
            result = self.oper.afft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        elif key == 'h':
            eta = self.state_phys.get_var('eta')
            result = 1 + eta

        elif key == 'Floc':
            h = self.compute('h')
            ux = self.state_phys.get_var('ux')
            uy = self.state_phys.get_var('uy')
            result = np.sqrt((ux**2 + uy**2)/(self.sim.params.c2*h))

        else:
            to_print = 'Do not know how to compute "'+key+'".'
            if RAISE_ERROR:
                raise ValueError(to_print)
            else:
                if mpi.rank == 0:
                    print(to_print + '\nreturn an array of zeros.')

                result = self.oper.constant_arrayX(value=0.)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result

    def statefft_from_statephys(self):
        """Compute the state in Fourier space."""
        ux = self.state_phys.get_var('ux')
        uy = self.state_phys.get_var('uy')
        eta = self.state_phys.get_var('eta')
        self.state_fft.set_var('ux_fft', self.oper.fft2(ux))
        self.state_fft.set_var('uy_fft', self.oper.fft2(uy))
        self.state_fft.set_var('eta_fft', self.oper.fft2(eta))

    def statephys_from_statefft(self):
        """Compute the state in physical space."""
        ifft2 = self.oper.ifft2
        ux_fft = self.state_fft.get_var('ux_fft')
        uy_fft = self.state_fft.get_var('uy_fft')
        eta_fft = self.state_fft.get_var('eta_fft')
        self.state_phys.set_var('ux', ifft2(ux_fft))
        self.state_phys.set_var('uy', ifft2(uy_fft))
        self.state_phys.set_var('eta', ifft2(eta_fft))
        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        self.state_phys.set_var('rot', ifft2(rot_fft))

    def return_statephys_from_statefft(self, state_fft=None):
        """Return the state in physical space."""
        ifft2 = self.oper.ifft2
        if state_fft is None:
            state_fft = self.state_fft
        ux_fft = state_fft.get_var('ux_fft')
        uy_fft = state_fft.get_var('uy_fft')
        eta_fft = state_fft.get_var('eta_fft')
        state_phys = SetOfVariables(like=self.state_phys)
        state_phys.set_var('ux', ifft2(ux_fft))
        state_phys.set_var('uy', ifft2(uy_fft))
        state_phys.set_var('eta', ifft2(eta_fft))

        rot_fft = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)
        state_phys.set_var('rot', ifft2(rot_fft))

        return state_phys
    
    def init_fft_from(self, **kwargs):
        if len(kwargs) == 1:
            if 'q_fft' in kwargs:
                self.init_from_qfft(kwargs['q_fft'])
            if 'a_fft' in kwargs:
                self.init_from_afft(kwargs['a_fft'])
        else:
            super(StateSW1L, self).init_statefft_from(**kwargs)

    def init_from_etafft(self, eta_fft):
        state_fft = self.state_fft
        state_fft.set_var('ux_fft', np.zeros_like(eta_fft))
        state_fft.set_var('uy_fft', np.zeros_like(eta_fft))
        state_fft.set_var('eta_fft', eta_fft)

        self.oper.dealiasing(state_fft)
        self.statephys_from_statefft()

    def init_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft):
        state_fft = self.state_fft
        state_fft.set_var('ux_fft', ux_fft)
        state_fft.set_var('uy_fft', uy_fft)
        state_fft.set_var('eta_fft', eta_fft)

        self.oper.dealiasing(state_fft)
        self.statephys_from_statefft()

    def init_from_rotuxuyfft(self, rot, ux_fft, uy_fft):
        self.init_from_uxuyfft(ux_fft, uy_fft)

    def init_from_rotfft(self, rot_fft):
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        self.init_from_uxuyfft(ux_fft, uy_fft)

    def init_from_qfft(self, q_fft):
        rot_fft = self.oper.rotfft_from_qfft(q_fft)
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        eta_fft = self.oper.etafft_from_qfft(q_fft)
        self.init_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

    def init_from_afft(self, a_fft):
        ux_fft, uy_fft, eta_fft = self.oper.uxuyetafft_from_afft(a_fft)
        self.init_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

    def init_from_uxuyfft(self, ux_fft, uy_fft):
        oper = self.oper
        ifft2 = oper.ifft2

        oper.projection_perp(ux_fft, uy_fft)
        oper.dealiasing(ux_fft, uy_fft)

        ux = ifft2(ux_fft)
        uy = ifft2(uy_fft)

        rot_fft = oper.rotfft_from_vecfft(ux_fft, uy_fft)
        rot = ifft2(rot_fft)

        eta_fft = self._etafft_no_div(ux, uy, rot)
        eta = ifft2(eta_fft)

        state_fft = self.state_fft
        state_fft.set_var('ux_fft', ux_fft)
        state_fft.set_var('uy_fft', uy_fft)
        state_fft.set_var('eta_fft', eta_fft)

        state_phys = self.state_phys
        state_phys.set_var('rot', rot)
        state_phys.set_var('ux', ux)
        state_phys.set_var('uy', uy)
        state_phys.set_var('eta', eta)

    def _etafft_no_div(self, ux, uy, rot):
        K2_not0 = self.oper.K2_not0
        rot_abs = rot + self.params.f

        tempx_fft = - self.oper.fft2(rot_abs*uy)
        tempy_fft = self.oper.fft2(rot_abs*ux)

        uu2_fft = self.oper.fft2(ux**2 + uy**2)

        eta_fft = (1.j * self.oper.KX*tempx_fft/K2_not0 +
                   1.j*self.oper.KY*tempy_fft/K2_not0 -
                   uu2_fft/2)/self.params.c2
        if mpi.rank == 0:
            eta_fft[0, 0] = 0.
        self.oper.dealiasing(eta_fft)

        return eta_fft
