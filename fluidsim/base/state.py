"""State of the variables (:mod:`fluidsim.base.state`)
============================================================

.. currentmodule:: fluidsim.base.state

Provides:

.. autoclass:: StateBase
   :members:
   :private-members:

.. autoclass:: StatePseudoSpectral
   :members:
   :private-members:

"""

import numpy as np

from fluidsim.operators.setofvariables import SetOfVariables


class StateBase(object):
    """Contains the state variables and handles the access to fields."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ContainerXML info_solver.

        This is a static method!
        """
        info_solver.classes.State.set_attribs(
            {'keys_state_phys': ['ux', 'uy'],
             'keys_computable': [],
             'keys_phys_needed': ['ux', 'uy']})

    def __init__(self, sim, info_solver):
        self.sim = sim
        self.params = sim.params
        self.oper = sim.oper

        # creation of the SetOfVariables state_fft and state_phys
        self.keys_state_phys = info_solver.classes.State.keys_state_phys
        self.keys_computable = info_solver.classes.State.keys_computable

        self.state_phys = SetOfVariables(keys=self.keys_state_phys,
                                         shape1var=self.oper.shapeX_loc,
                                         dtype=np.float64,
                                         name_type_variables='state_phys'
                                         )
        self.vars_computed = {}
        self.it_computed = {}

    def compute(self, key):
        pass

    def clear_computed(self):
        self.vars_computed.clear()

    def __call__(self, key):
        if key in self.keys_state_phys:
            return self.state_phys[key]
        else:
            it = self.sim.time_stepping.it
            if (key in self.vars_computed and it == self.it_computed[key]):
                return self.vars_computed[key]
            else:
                value = self.compute(key)
                self.vars_computed[key] = value
                self.it_computed[key] = it
                return value

    def __setitem__(self, key, value):
        if key in self.keys_state_phys:
            self.state_phys[key] = value
        else:
            raise ValueError('key "'+key+'" is not known')

    def can_this_key_be_obtained(self, key):
        return (key in self.keys_state_phys or
                key in self.keys_computable)


class StatePseudoSpectral(StateBase):
    """Contains the state variables and handles the access to fields.

    This is the general class for the pseudo-spectral solvers.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ContainerXML info_solver.

        This is a static method!
        """

        StateBase._complete_info_solver(info_solver)

        info_solver.classes.State.set_attribs(
            {'keys_state_fft': ['ux_fft', 'uy_fft']})

    def __init__(self, sim, info_solver):

        super(StatePseudoSpectral, self).__init__(sim, info_solver)

        self.keys_state_fft = info_solver.classes.State['keys_state_fft']
        self.state_fft = SetOfVariables(keys=self.keys_state_fft,
                                        shape1var=self.oper.shapeK_loc,
                                        dtype=np.complex128,
                                        name_type_variables='state_fft')

    def __call__(self, key):
        """Return the variable corresponding to the given key."""
        if key in self.keys_state_fft:
            return self.state_fft[key]
        elif key in self.keys_state_phys:
            return self.state_phys[key]
        else:
            it = self.sim.time_stepping.it
            if (key in self.vars_computed and it == self.it_computed[key]):
                return self.vars_computed[key]
            else:
                value = self.compute(key)
                self.vars_computed[key] = value
                self.it_computed[key] = it
                return value

    def __setitem__(self, key, value):
        if key in self.keys_state_fft:
            self.state_fft[key] = value
        elif key in self.keys_state_phys:
            self.state_phys[key] = value
        else:
            raise ValueError('key "'+key+'" is not known')

    def statefft_from_statephys(self):
        fft2 = self.oper.fft2
        for ik in xrange(self.state_fft.nb_variables):
            self.state_fft.data[ik][:] = fft2(self.state_phys.data[ik])

    def statephys_from_statefft(self):
        ifft2 = self.oper.ifft2
        for ik in xrange(self.state_fft.nb_variables):
            self.state_phys.data[ik] = ifft2(self.state_fft.data[ik])

    def return_statephys_from_statefft(self, state_fft=None):
        """Return the state in physical space."""
        ifft2 = self.oper.ifft2
        if state_fft is None:
            state_fft = self.state_fft

        state_phys = SetOfVariables(like_this_sov=self.state_phys)
        for ik in xrange(self.state_fft.nb_variables):
            state_phys.data[ik] = ifft2(state_fft.data[ik])
        return state_phys

    def can_this_key_be_obtained(self, key):
        return (key in self.keys_state_phys or
                key in self.keys_computable or
                key in self.keys_state_fft)
