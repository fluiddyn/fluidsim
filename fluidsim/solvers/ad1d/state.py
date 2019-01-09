"""State for the AD1D solver (:mod:`fluidsim.solvers.ad1d.state`)
=======================================================================
"""

from fluidsim.base.state import StateBase

from fluiddyn.util import mpi


class StateAD1D(StateBase):
    """Contains the variables corresponding to the state and handles the
    access to other fields for the solver AD1D.

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {
                "keys_state_phys": ["s"],
                "keys_computable": [],
                "keys_phys_needed": ["s"],
                "keys_linear_eigenmodes": ["s"],
            }
        )

    def compute(self, key, SAVE_IN_DICT=True):
        it = self.sim.time_stepping.it
        if key in self.vars_computed and it == self.it_computed[key]:
            return self.vars_computed[key]

        if key == "dx_s":
            result = self.oper.px(self.state_phys.get_var("s"))

        else:
            raise ValueError('Do not know how to compute "' + key + '".')

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

        return result
