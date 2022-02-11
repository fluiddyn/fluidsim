"""State of the variables (:mod:`fluidsim.base.basilisk.state`)
===============================================================

Provides:

.. autoclass:: StateBasilisk
   :members:
   :private-members:

"""


from ..state import StateBase


class StateBasilisk(StateBase):
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {
                "keys_state_phys": ["ux", "uy", "rot"],
                "keys_computable": [],
                "keys_phys_needed": ["rot"],
            }
        )

    def compute(self, key):
        """Compute scalar fields such a component of the velocity or vorticity."""
        if key == "ux":
            scalar = self.sim.basilisk.uf.x
        elif key == "uy":
            scalar = self.sim.basilisk.uf.y
        elif key == "rot":
            scalar = self.sim.basilisk.omega
        else:
            raise ValueError('No method to compute key "' + key + '"')

        return scalar.f(self.oper.X, self.oper.Y)

    def _get_state_from_basilisk(self):
        for key in self.keys_state_phys:
            self.state_phys.set_var(key, self.compute(key))
