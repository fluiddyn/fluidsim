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

import numpy as np

from fluidsim.base.setofvariables import SetOfVariables


class StateBase:
    """Contains the state variables and handles the access to fields.

    Parameters
    ----------

    sim : child of :class:`fluidsim.base.solvers.base.SimulBase`

    oper : Optional[operators]

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {
                "keys_state_phys": ["X"],
                "keys_computable": [],
                "keys_phys_needed": ["X"],
            }
        )

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
            self.keys_computable = sim.info.solver.classes.State.keys_computable
        except AttributeError:
            self.keys_computable = []

        self.state_phys = SetOfVariables(
            keys=self.keys_state_phys,
            shape_variable=self.oper.shapeX_loc,
            dtype=np.float64,
            info="state_phys",
        )
        self.vars_computed = {}
        self.it_computed = {}

        self.is_initialized = False

    def compute(self, key):
        """Compute a not stored variable from the stored variables"""
        raise ValueError('No method to compute key "' + key + '"')

    def clear_computed(self):
        """Clear the stored computed variables."""
        self.vars_computed.clear()

    def has_vars(self, *keys):
        """Checks if all of the keys are present in the union of
        ``keys_state_phys`` and ``keys_computable``.

        Parameters
        ----------
        keys: str, str ...
            Strings indicating state variable names.

        Returns
        -------
        bool

        Examples
        --------
        >>> sim.state.has_vars('ux', 'uy')
        >>> sim.state.has_vars('ux')
        >>> sim.state.has_vars('ux', 'vx', strict=False)

        .. todo::

           ``strict=True`` can be a Python 3 compatible keywords-only argument
           with the function like::

               def has_vars(self, *keys, strict=True):
                   ...
                   if strict:
                       return keys.issubset(keys_state)
                   else:
                       return len(keys.intersection(keys_state)) > 0

            When ``True``, checks if all keys form a subset of state keys. When
            ``False``, checks if the intersection of the keys and the state keys
            has atleast one member.

        """
        keys_state = set(self.keys_state_phys + self.keys_computable)
        keys = set(keys)
        return keys.issubset(keys_state)

    def get_var(self, key):
        """Get a physical variable (from the storage array or computed).

        This is one of the main method of the state classes.

        It tries to return the array corresponding to a physical variable. If
        it is stored in the main storage array of the state class, it is
        directly returned.  Otherwise, we try to compute the quantity with the
        method :func:`compute`.

        It should not be necessary to redefine this method in child class.

        """
        if key in self.keys_state_phys:
            return self.state_phys.get_var(key)

        else:
            it = self.sim.time_stepping.it
            if key in self.vars_computed and it == self.it_computed[key]:
                return self.vars_computed[key]

            else:
                value = self.compute(key)
                self.vars_computed[key] = value
                self.it_computed[key] = it
                return value

    def __call__(self, key):
        raise DeprecationWarning(
            "Do not call a state object. " "Instead, use get_var method."
        )

    def __setitem__(self, key, value):
        """General setter function to set the value of a variable

        It should not be necessary to redefine this method in child class.
        """
        if key in self.keys_state_phys:
            self.state_phys.set_var(key, value)
        else:
            raise ValueError('key "' + key + '" is not known')

    def can_this_key_be_obtained(self, key):
        """To check whether a variable can be obtained.

        .. deprecated:: 0.2.0
           Use ``has_vars`` method instead.

        """
        raise DeprecationWarning(
            "Do not call can_this_key_be_obtained. "
            "Instead, use has_vars method."
        )

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
        self.state_phys[:] = 0.0

        for key, value in list(kwargs.items()):
            if key not in self.keys_state_phys:
                raise ValueError(
                    f'Do not know how to initialize with key "{key}".'
                )

            self.state_phys.set_var(key, value)


class StatePseudoSpectral(StateBase):
    """Contains the state variables and handles the access to fields.

    This is the general class for the pseudo-spectral solvers.

    Parameters
    ----------

    sim : child of :class:`fluidsim.base.solvers.base.SimulBase`

    oper : Optional[operators]

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """

        StateBase._complete_info_solver(info_solver)

        info_solver.classes.State.keys_state_phys = ["ux", "uy"]
        info_solver.classes.State.keys_phys_needed = ["ux", "uy"]

        info_solver.classes.State._set_attribs(
            {"keys_state_spect": ["ux_fft", "uy_fft"]}
        )

    def __init__(self, sim, oper=None):

        super().__init__(sim, oper)

        self.keys_state_spect = sim.info.solver.classes.State.keys_state_spect
        self.state_spect = SetOfVariables(
            keys=self.keys_state_spect,
            shape_variable=self.oper.shapeK_loc,
            dtype=np.complex128,
            info="state_spect",
        )

    def has_vars(self, *keys):
        """Checks if all of the keys are present in the union of
        ``keys_state_phys``, ``keys_computable``, and ``keys_state_spect``.

        Parameters
        ----------
        keys: str, str ...
            Strings indicating state variable names.

        Returns
        -------
        bool

        Examples
        --------
        >>> sim.state.has_vars('ux', 'uy', 'ux_fft')
        >>> sim.state.has_vars('rot')

        """
        keys_state = set(
            self.keys_state_phys + self.keys_computable + self.keys_state_spect
        )
        keys = set(keys)
        return keys.issubset(keys_state)

    def get_var(self, key):
        """Get a variable (from the storage arrays or computed).

        This is one of the main method of the state classes.

        It tries to return the array corresponding to a physical variable. If
        it is stored in the main storage arrays (`state_phys` and `state_spec`)
        of the state class, it is directly returned.  Otherwise, we try to
        compute the quantity with the method :func:`compute`.

        It should not be necessary to redefine this method in child class.

        """

        if key in self.keys_state_spect:
            return self.state_spect.get_var(key)

        elif key in self.keys_state_phys:
            return self.state_phys.get_var(key)

        else:
            it = self.sim.time_stepping.it
            if key in self.vars_computed and it == self.it_computed[key]:
                return self.vars_computed[key]

            else:
                value = self.compute(key)
                self.vars_computed[key] = value
                self.it_computed[key] = it
                return value

    def __setitem__(self, key, value):
        """General setter function to set the value of a variable

        It should not be necessary to redefine this method in child class.
        """
        if key in self.keys_state_spect:
            self.state_spect.set_var(key, value)
        elif key in self.keys_state_phys:
            self.state_phys.set_var(key, value)
        else:
            raise ValueError('key "' + key + '" is not known')

    def statespect_from_statephys(self):
        """Compute the spectral variables from the physical variables.

        When you implement a new solver, check that this method does the job!
        """
        for ik in range(self.state_spect.nvar):
            self.oper.fft_as_arg(self.state_phys[ik], self.state_spect[ik])

    def statephys_from_statespect(self):
        """Compute the physical variables from the spectral variables.

        When you implement a new solver, check that this method does the job!
        """
        for ik in range(self.state_spect.nvar):
            self.oper.ifft_as_arg(self.state_spect[ik], self.state_phys[ik])

    def return_statephys_from_statespect(self, state_spect=None):
        """Return the physical variables computed from the spectral variables."""
        ifft = self.oper.ifft
        if state_spect is None:
            state_spect = self.state_spect

        state_phys = SetOfVariables(like=self.state_phys)
        for ik in range(self.state_spect.nvar):
            state_phys[ik] = ifft(state_spect[ik])
        return state_phys

    def can_this_key_be_obtained(self, key):
        return (
            key in self.keys_state_phys
            or key in self.keys_computable
            or key in self.keys_state_spect
        )

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
        self.state_spect[:] = 0.0

        for key, value in list(kwargs.items()):
            if key not in self.keys_state_spect:
                raise ValueError(
                    f'Do not know how to initialize with key "{key}".'
                )

            self.state_spect.set_var(key, value)

    def check_energy_equal_phys_spect(self):
        energy_spect = self.sim.output.compute_energy()
        energy_phys = self.compute_energy_phys()
        if not np.allclose(energy_spect, energy_phys):
            raise RuntimeError(
                "Physical and spectral states are inconsistent: "
                f"{energy_spect} != {energy_phys}"
            )

    def compute_energy_phys(self):
        raise NotImplementedError
