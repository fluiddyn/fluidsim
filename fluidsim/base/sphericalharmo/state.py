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

from ..setofvariables import SetOfVariables
from ..state import StatePseudoSpectral


class StateSphericalHarmo(StatePseudoSpectral):
    """Contains the state variables and handles the access to fields.

    This is the general class for the pseudo-spectral solvers.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """

        StatePseudoSpectral._complete_info_solver(info_solver)

        info_solver.classes.State._set_attribs({
            'keys_state_spect': ['rot_sh'],
            'keys_state_phys': ['ux', 'uy', 'rot'],
            'keys_computable': [],
            'keys_phys_needed': ['rot'],
            'keys_linear_eigenmodes': ['rot_sh']})

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        """Compute and return a variable"""
        it = self.sim.time_stepping.it
        if (key in self.vars_computed and it == self.it_computed[key]):
            return self.vars_computed[key]

        results = {}

        if key == 'ux_sh':
            result = self.oper.sht(self.state_phys.get_var('ux'))
        elif key == 'uy_sh':
            result = self.oper.sht(self.state_phys.get_var('uy'))
        elif key == 'rot_sh':
            ux = self.state_phys.get_var('ux')
            uy = self.state_phys.get_var('uy')
            div_sh, rot_sh = self.oper.hdivrotsh_from_uv(ux, uy)
            result = rot_sh
            results = {'div_sh': div_sh}
        elif key == 'div_sh':
            ux = self.state_phys.get_var('ux')
            uy = self.state_phys.get_var('uy')
            div_sh, rot_sh = self.oper.hdivrotsh_from_uv(ux, uy)
            result = div_sh
            results = {'rot_sh': rot_sh}
        elif key == 'rot':
            result = self.state_phys.get_var('rot')
        elif key == 'div':
            result = self.oper.create_array_spat(0.)
        elif key == 'q':
            rot = self.state_phys.get_var('rot')
            result = rot
        else:
            to_print = 'Do not know how to compute "' + key + '".'
            if RAISE_ERROR:
                raise ValueError(to_print)
            else:
                print(to_print +
                      '\nreturn an array of zeros.')

                result = self.oper.constant_arrayX(value=0.)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

            for key, var in results.items():
                self.vars_computed[key] = var
                self.it_computed[key] = it

        return result

    def statespect_from_statephys(self):
        for ik in range(self.state_spect.nvar):
            self.oper.sht_as_arg(self.state_phys[ik], self.state_spect[ik])

    def statephys_from_statespect(self):
        for ik in range(self.state_spect.nvar):
            self.oper.isht_as_arg(self.state_spect[ik], self.state_phys[ik])

    def return_statephys_from_statespect(self, state_spect=None):
        """Return the state in physical space."""
        isht = self.oper.isht
        if state_spect is None:
            state_spect = self.state_spect

        state_phys = SetOfVariables(like=self.state_phys)
        for ik in range(self.state_spect.nvar):
            state_phys[ik] = isht(state_spect[ik])
        return state_phys
