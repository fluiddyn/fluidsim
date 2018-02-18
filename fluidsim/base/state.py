"""State of the variables (:mod:`fluidsim.base.state`)
============================================================

Provides:

.. autoclass:: StateBase
   :members:
   :private-members:

.. autoclass:: StatePseudoSpectral
   :members:
   :private-members:

"""

from builtins import range
from builtins import object
import numpy as np

from fluidsim.base.setofvariables import SetOfVariables


class StateBase(object):
    """Contains the state variables and handles the access to fields."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {'keys_state_phys': ['X'],
             'keys_computable': [],
             'keys_phys_needed': ['X']})

    def __init__(self, sim, oper=None):
        self.sim = sim
        self.params = sim.params
        if oper is None:
            self.oper = sim.oper
        else:
            self.oper = oper

        # creation of the SetOfVariables state_spect and state_phys
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
        raise ValueError('No method to compute key "' + key + '"')

    def clear_computed(self):
        self.vars_computed.clear()

    def get_var(self, key):
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

    def __call__(self, key):
        raise DeprecationWarning('Do not call a state object. '
                                 'Instead, use its get_var method.')

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

        info_solver.classes.State.keys_state_phys = ['ux', 'uy']
        info_solver.classes.State.keys_phys_needed = ['ux', 'uy']

        info_solver.classes.State._set_attribs(
            {'keys_state_spect': ['ux_fft', 'uy_fft']})

    def __init__(self, sim, oper=None):

        super(StatePseudoSpectral, self).__init__(sim, oper)

        self.keys_state_spect = sim.info.solver.classes.State.keys_state_spect
        self.state_spect = SetOfVariables(keys=self.keys_state_spect,
                                          shape_variable=self.oper.shapeK_loc,
                                          dtype=np.complex128,
                                          info='state_spect')

    def get_var(self, key):
        """Return the variable corresponding to the given key."""

        if key in self.keys_state_spect:
            return self.state_spect.get_var(key)
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
        if key in self.keys_state_spect:
            self.state_spect.set_var(key, value)
        elif key in self.keys_state_phys:
            self.state_phys.set_var(key, value)
        else:
            raise ValueError('key "'+key+'" is not known')

    def statespect_from_statephys(self):
        for ik in range(self.state_spect.nvar):
            self.oper.fft_as_arg(self.state_phys[ik], self.state_spect[ik])

    def statephys_from_statespect(self):
        for ik in range(self.state_spect.nvar):
            self.oper.ifft_as_arg(self.state_spect[ik], self.state_phys[ik])

    def return_statephys_from_statespect(self, state_spect=None):
        """Return the state in physical space."""
        ifft2 = self.oper.ifft2
        if state_spect is None:
            state_spect = self.state_spect

        state_phys = SetOfVariables(like=self.state_phys)
        for ik in range(self.state_spect.nvar):
            state_phys[ik] = ifft2(state_spect[ik])
        return state_phys

    def can_this_key_be_obtained(self, key):
        return (key in self.keys_state_phys or
                key in self.keys_computable or
                key in self.keys_state_spect)

    def init_statespect_from(self, **kwargs):
        """Initialize `state_spect` from arrays.

        Parameters
        ----------

        **kwargs : {key: array, ...}

          keys and arrays used for the initialization. The other keys
          are set to zero.

        Examples
        --------

        .. code-block:: python

           kwargs = {'a_fft': Fa_fft}
           init_statespect_from(**kwargs)

           ux_fft, uy_fft, eta_fft = oper.uxuyetafft_from_qfft(q_fft)
           init_statespect_from(ux_fft=ux_fft, uy_fft=uy_fft, eta_fft=eta_fft)

        """
        self.state_spect[:] = 0.

        for key, value in list(kwargs.items()):
            if key not in self.keys_state_spect:
                raise ValueError(
                    'Do not know how to initialize with key "{}".'.format(key))
            self.state_spect.set_var(key, value)

    def init_statephys_from(self, **kwargs):
        """Initialize `state_phys` from arrays.

        Parameters
        ----------

        **kwargs : {key: array, ...}

          keys and arrays used for the initialization. The other keys
          are set to zero.

        Examples
        --------

        .. code-block:: python

           kwargs = {'a': Fa}
           init_statespect_from(**kwargs)

           init_statespect_from(ux=ux, uy=uy, eta=eta)

        """
        self.state_phys[:] = 0.

        for key, value in list(kwargs.items()):
            if key not in self.keys_state_phys:
                raise ValueError(
                    'Do not know how to initialize with key "{}".'.format(key))
            self.state_phys.set_var(key, value)
