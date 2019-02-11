"""State of the variables (:mod:`fluidsim.base.sphericalharmo.state`)
=====================================================================

Provides:

.. autoclass:: StateSphericalHarmo
   :members:
   :private-members:

"""

# from ..setofvariables import SetOfVariables
from ..state import StatePseudoSpectral


class StateSphericalHarmo(StatePseudoSpectral):
    """Contains the state variables and handles the access to fields.

    This is the general class for the pseudo-spectral solvers base on the
    Sperical harmonic transform.

    .. warning::

       We assume incompressibility (div = 0) in this base class!

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Static method to complete the ParamContainer info_solver."""

        StatePseudoSpectral._complete_info_solver(info_solver)

        info_solver.classes.State._set_attribs(
            {
                "keys_state_spect": ["rot_sh"],
                "keys_state_phys": ["ux", "uy", "rot"],
                "keys_computable": [],
                "keys_phys_needed": ["rot"],
                "keys_linear_eigenmodes": ["rot_sh"],
            }
        )

    def compute(self, key, SAVE_IN_DICT=True, RAISE_ERROR=True):
        """Compute and return a variable"""
        it = self.sim.time_stepping.it
        if key in self.vars_computed and it == self.it_computed[key]:
            return self.vars_computed[key]

        results = {}

        if key == "ux_sh":
            result = self.oper.sht(self.state_phys.get_var("ux"))
        elif key == "uy_sh":
            result = self.oper.sht(self.state_phys.get_var("uy"))
        elif key == "div_sh":
            result = self.oper.create_array_sh(0.0)
        elif key == "div":
            result = self.oper.create_array_spat(0.0)
        elif key == "q":
            rot = self.state_phys.get_var("rot")
            result = rot
        else:
            to_print = 'Do not know how to compute "' + key + '".'
            if RAISE_ERROR:
                raise ValueError(to_print)

            else:
                print(to_print + "\nreturn an array of zeros.")

                result = self.oper.create_arrayX(value=0.0)

        if SAVE_IN_DICT:
            self.vars_computed[key] = result
            self.it_computed[key] = it

            for key, var in results.items():
                self.vars_computed[key] = var
                self.it_computed[key] = it

        return result

    def statephys_from_statespect(self):
        """Compute `state_phys` from `statespect`."""
        rot_sh = self.state_spect.get_var("rot_sh")

        # efficient
        rot = self.state_phys.get_var("rot")
        self.oper.isht_as_arg(rot_sh, rot)

        # not very efficient!
        ux, uy = self.oper.vec_from_rotsh(rot_sh)
        self.state_phys.set_var("ux", ux)
        self.state_phys.set_var("uy", uy)

    def statespect_from_statephys(self):
        """Compute `state_spect` from `state_phys`."""

        rot = self.state_phys.get_var("rot")
        rot_sh = self.state_spect.get_var("rot_sh")
        self.oper.sht_as_arg(rot, rot_sh)

    def init_from_rotsh(self, rot_sh):
        """Initialize the state from the variable `rot_sh`."""
        self.oper.dealiasing(rot_sh)
        self.state_spect.set_var("rot_sh", rot_sh)
        self.statephys_from_statespect()

    def init_statespect_from(self, **kwargs):
        if len(kwargs) == 1:
            if "rot_sh" in kwargs:
                self.init_from_rotsh(kwargs["rot_sh"])
        else:
            super().init_statespect_from(**kwargs)
