"""State of the variables (:mod:`fluidsim.base.sphericalharmo.state`)
=====================================================================

Provides:

.. autoclass:: StateSphericalHarmoSW1L
   :members:
   :private-members:

"""
from fluidsim.base.state import StatePseudoSpectral
from fluidsim.base.setofvariables import SetOfVariables


class StateSphericalHarmoSW1L(StatePseudoSpectral):
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
                "keys_state_spect": ["rot_sh", "div_sh", "eta_sh"],
                "keys_state_phys": ["ux", "uy", "eta", "rot"],
                "keys_computable": ["div", "q"],
                "keys_phys_needed": [],
                "keys_linear_eigenmodes": [],
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
        elif key == "div":
            result = self.oper.isht(self.state_spect.get_var("div_sh"))
        elif key == "q":
            rot = self.state_phys.get_var("rot")
            eta = self.state_phys.get_var("eta")
            result = rot - self.oper.f_radial * eta
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

    def statephys_from_statespect(self, state_spect=None, state_phys=None):
        """Compute the state in physical space."""
        if state_spect is None:
            state_spect = self.state_spect

        if state_phys is None:
            state_phys = self.state_phys

        rot_sh = state_spect.get_var("rot_sh")
        div_sh = state_spect.get_var("div_sh")
        eta_sh = state_spect.get_var("eta_sh")

        ux = state_phys.get_var("ux")
        uy = state_phys.get_var("uy")
        eta = state_phys.get_var("eta")
        rot = state_phys.get_var("rot")

        self.oper.vec_from_divrotsh(div_sh, rot_sh, ux, uy)
        self.oper.isht_as_arg(eta_sh, eta)
        self.oper.isht_as_arg(rot_sh, rot)

    def return_statephys_from_statespect(self, state_spect=None):
        """Return the state in physical space as a new object separate from
        ``self.state_phys``.

        """
        state_phys = SetOfVariables(like=self.state_phys)
        self.statephys_from_statespect(state_spect, state_phys)
        return state_phys

    def statespect_from_statephys(self):
        """Compute `state_spect` from `state_phys`."""
        ux = self.state_phys.get_var("ux")
        uy = self.state_phys.get_var("uy")
        eta = self.state_phys.get_var("eta")

        div_sh = self.state_spect.get_var("div_sh")
        rot_sh = self.state_spect.get_var("rot_sh")
        eta_sh = self.state_spect.get_var("eta_sh")
        self.oper.divrotsh_from_vec(ux, uy, div_sh, rot_sh)
        self.oper.sht_as_arg(eta, eta_sh)

    def init_from_uxuyeta(self, ux, uy, eta):
        """Initialize from ``ux``, ``uy`` the velocities and ``eta`` the
        displacement field.

        """
        self.state_phys.set_var("ux", ux)
        self.state_phys.set_var("uy", uy)
        self.state_phys.set_var("eta", eta)

        self.statespect_from_statephys()

        rot_sh = self.state_spect.get_var("rot_sh")
        rot = self.state_phys.get_var("rot")
        self.oper.isht_as_arg(rot_sh, rot)

    def init_from_rotsh(self, rot_sh):
        """Initialize the state from the variable `rot_sh`."""
        # self.oper.dealiasing(rot_sh) -> pointless
        self.state_spect.set_var("rot_sh", rot_sh)
        self.statephys_from_statespect()

    def init_statespect_from(self, **kwargs):
        if len(kwargs) == 1:
            if "rot_sh" in kwargs:
                self.init_from_rotsh(kwargs["rot_sh"])
        else:
            super().init_statespect_from(**kwargs)
