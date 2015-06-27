"""State for the NS2D solver (:mod:`fluidsim.solvers.ns2d.state`)
=================================================================

.. autoclass:: StateNS2D
   :members:
   :private-members:

"""


from fluidsim.base.state import StatePseudoSpectral

from fluiddyn.util import mpi


class StateNS2D(StatePseudoSpectral):
    """State for the solver ns2d.

    Contains the variables corresponding to the state and handles the
    access to other fields.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""
        info_solver.classes.State._set_attribs({
            'keys_state_fft': ['rot_fft'],
            'keys_state_phys': ['ux', 'uy', 'rot'],
            'keys_computable': [],
            'keys_phys_needed': ['rot'],
            'keys_linear_eigenmodes': ['rot_fft']})

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
        self.state_phys.set_var('rot', self.oper.ifft2(rot_fft))
        ux_fft, uy_fft = self.oper.vecfft_from_rotfft(rot_fft)
        self.state_phys.set_var('ux', self.oper.ifft2(ux_fft))
        self.state_phys.set_var('uy', self.oper.ifft2(uy_fft))

    def statefft_from_statephys(self):
        """Compute `state_fft` from `state_phys`."""
        rot = self.state_phys.get_var('rot')
        self.state_fft.set_var('rot_fft', self.oper.fft2(rot))

    def init_from_rotfft(self, rot_fft):
        """Initialize the state from the variable `rot_fft`."""
        self.sim.oper.dealiasing(rot_fft)
        self.state_fft.set_var('rot_fft', rot_fft)
        self.statephys_from_statefft()
