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
from copy import copy

from fluidsim.base.setofvariables import SetOfVariables


class StateBase(object):
    """Contains the state variables and handles the access to fields."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {'keys_state_phys': ['ux', 'uy'],
             'keys_computable': [],
             'keys_phys_needed': ['ux', 'uy']})

    def __init__(self, sim, oper=None):
        self.sim = sim
        self.params = sim.params
        if oper is None:
            self.oper = sim.oper
        else:
            self.oper = oper

        # creation of the SetOfVariables state_fft and state_phys
        self.keys_state_phys = sim.info.solver.classes.State.keys_state_phys

        try:
            self.keys_computable = \
                sim.info.solver.classes.State.keys_computable
        except AttributeError:
            self.keys_computable = []

        self.state_phys = SetOfVariables(keys=self.keys_state_phys,
                                         shape_variable=self.oper.shapeX_loc,
                                         dtype=np.float64,
                                         info='state_phys')
        self.vars_computed = {}
        self.it_computed = {}

        self.is_initialized = False

    def compute(self, key):
        pass

    def clear_computed(self):
        self.vars_computed.clear()

    def __call__(self, key):
        if key in self.keys_state_phys:
            return self.state_phys.get_var(key)
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
            self.state_phys.set_var(key, value)
        else:
            raise ValueError('key "' + key + '" is not known')

    def can_this_key_be_obtained(self, key):
        return (key in self.keys_state_phys or
                key in self.keys_computable)


class StatePseudoSpectral(StateBase):
    """Contains the state variables and handles the access to fields.

    This is the general class for the pseudo-spectral solvers.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """

        StateBase._complete_info_solver(info_solver)

        info_solver.classes.State._set_attribs(
            {'keys_state_fft': ['ux_fft', 'uy_fft']})

    def __init__(self, sim, oper=None):

        super(StatePseudoSpectral, self).__init__(sim, oper)

        self.keys_state_fft = sim.info.solver.classes.State.keys_state_fft
        self.state_fft = SetOfVariables(keys=self.keys_state_fft,
                                        shape_variable=self.oper.shapeK_loc,
                                        dtype=np.complex128,
                                        info='state_fft')

    def __call__(self, key):
        """Return the variable corresponding to the given key."""
        if key in self.keys_state_fft:
            return self.state_fft.get_var(key)
        elif key in self.keys_state_phys:
            return self.state_phys.get_var(key)
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
            self.state_fft.set_var(key, value)
        elif key in self.keys_state_phys:
            self.state_phys.set_var(key, value)
        else:
            raise ValueError('key "'+key+'" is not known')

    def statefft_from_statephys(self):
        fft2 = self.oper.fft2
        for ik in xrange(self.state_fft.nvar):
            self.state_fft[ik][:] = fft2(self.state_phys[ik])

    def statephys_from_statefft(self):
        ifft2 = self.oper.ifft2
        for ik in xrange(self.state_fft.nvar):
            self.state_phys[ik] = ifft2(self.state_fft[ik])

    def return_statephys_from_statefft(self, state_fft=None):
        """Return the state in physical space."""
        ifft2 = self.oper.ifft2
        if state_fft is None:
            state_fft = self.state_fft

        state_phys = SetOfVariables(like=self.state_phys)
        for ik in xrange(self.state_fft.nvar):
            state_phys[ik] = ifft2(state_fft[ik])
        return state_phys

    def can_this_key_be_obtained(self, key):
        return (key in self.keys_state_phys or
                key in self.keys_computable or
                key in self.keys_state_fft)

    def init_statefft_from(self, **kwargs):
        """Initialize `state_fft` from arrays.

        Parameters
        ----------

        **kwargs : {key: array, ...}

          keys and arrays used for the initialization. The other keys
          are set to zero.

        Examples
        --------

        .. code-block:: python

           kwargs = {'a_fft': Fa_fft}
           init_statefft_from(**kwargs)

           ux_fft, uy_fft, eta_fft = oper.uxuyetafft_from_qfft(q_fft)
           init_statefft_from(ux_fft=ux_fft, uy_fft=uy_fft, eta_fft=eta_fft)

        """
        self.state_fft[:] = 0.

        for key, value in kwargs.items():
            if key not in self.keys_state_fft:
                raise ValueError(
                    'Do not know how to initialize with key "{}".'.format(key))
            self.state_fft.set_var(key, value)
