"""State for the NS2D solver (:mod:`fluidsim.solvers.ns2d.strat.state`)
=======================================================================

.. autoclass:: StateNS2DStrat
   :members:
   :private-members:

"""

import numpy as np

# from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi

from fluidsim.solvers.ns2d.state import StateNS2D


class StateNS2DStrat(StateNS2D):
    """..."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Update `info_solver` container with the stratification terms."""
        # Updating the state to a stratified state
        info_solver.classes.State._set_attribs({
            'keys_state_fft': ['rot_fft', 'b_fft'],
            'keys_state_phys': ['ux', 'uy', 'rot', 'b'],
            'keys_computable': [],
            'keys_phys_needed': ['rot', 'b'],
            'keys_linear_eigenmodes': ['rot_fft', 'b_fft']})

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        """Compute and return a variable"""
        it = self.sim.time_stepping.it
        if (key in self.vars_computed and it == self.it_computed[key]):
            return self.vars_computed[key]

        if key == 'ux_fft':
            result = self.oper.fft2(self.state_phys.get_var('ux'))
        elif key == 'uy_fft':
            result = self.oper.fft2(self.state_phys.get_var('uy'))
        elif key == 'rot_fft':
            ux_fft = self.compute('ux_fft')
            uy_fft = self.compute('uy_fft')
            result = self.oper.rotfft_from_vecfft(ux_fft, uy_fft)

        # Introduction buoyancy term 'b'
        elif key == 'b_fft':
            result = self.oper.fft2(self.state_phys.get_var('b'))

        elif key == 'div_fft':
            ux_fft = self.compute('ux_fft')
            uy_fft = self.compute('uy_fft')
            result = self.oper.divfft_from_vecfft(ux_fft, uy_fft)
        elif key == 'rot':
            rot_fft = self.compute('rot_fft')
            result = self.oper.ifft2(rot_fft)
        elif key == 'div':
            div_fft = self.compute('div_fft')
            result = self.oper.ifft2(div_fft)
        elif key == 'q':
            rot = self.compute('rot')
            result = rot
        else:
            to_print = 'Do not know how to compute "' + key + '".'
            if RAISE_ERROR:
                raise ValueError(to_print)
            else:
                if mpi.rank == 0:
                    print(to_print +
                          '\nreturn an array of zeros.')

                result = self.oper.constant_arrayX(value=0.)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result

    def statephys_from_statefft(self):
        """Compute `state_phys` from `statefft`."""
        rot_fft = self.state_fft.get_var('rot_fft')

        # Compute b from b_fft
        b_fft = self.state_fft.get_var('b_fft')
        self.state_phys.set_var('b', self.oper.ifft2(b_fft))

        self.state_phys.set_var('rot', self.oper.ifft2(rot_fft))
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        self.state_phys.set_var('ux', self.oper.ifft2(ux_fft))
        self.state_phys.set_var('uy', self.oper.ifft2(uy_fft))

    def statefft_from_statephys(self):
        """Compute `state_fft` from `state_phys`."""

        # Compute b_fft from b
        b = self.state_phys.get_var('b')
        self.state_fft.set_var('b_fft', self.oper.fft2(b))

        rot = self.state_phys.get_var('rot')
        self.state_fft.set_var('rot_fft', self.oper.fft2(rot))

    def init_from_rotbfft(self, rot_fft, b_fft):
        """Initialize the state from the variable `rot_fft`."""
        self.oper.dealiasing(rot_fft)
        self.oper.dealiasing(b_fft)

        self.state_fft.set_var('rot_fft', rot_fft)
        self.state_fft.set_var('b_fft', b_fft)

        self.statephys_from_statefft()

    def init_from_rotfft(self, rot_fft):
        b_fft = np.zeros(self.oper.shapeK_loc, dtype=np.complex128)
        self.init_from_rotbfft(rot_fft, b_fft)

    def init_fft_from(self, **kwargs):

        # init_fft_from looks if kwargs has two arguments.
        if len(kwargs) == 2:
            if 'rot_fft' in kwargs and 'b_fft' in kwargs:
                self.init_from_rotbfft(kwargs['rot_fft'], kwargs['b_fft'])
        else:
            super(StateNS2D, self).init_statefft_from(**kwargs)
