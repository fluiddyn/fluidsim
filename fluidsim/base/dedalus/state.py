"""State of the variables (:mod:`fluidsim.base.dedalus.state`)
===============================================================

Provides:

.. autoclass:: StateDedalus
   :members:
   :private-members:

"""
import numpy as np

from ..state import StateBase


class StatePhysDedalus:
    def __init__(self, dedalus_solver, keys):
        self.dedalus_solver = dedalus_solver
        self.keys = keys

    def get_var(self, key):
        if self.dedalus_solver is None:
            return np.zeros(1)

        field = self.dedalus_solver.state[key]
        field.set_scales(1.0)
        return field["g"].transpose()

    def set_var(self, key, value):
        field = self.dedalus_solver.state[key]
        field.set_scales(1.0)
        field["g"] = value.transpose()

    def initialize(self, value):
        for key in self.keys:
            self.get_var(key).fill(value)

    @property
    def nbytes(self):
        return 1

    @property
    def info(self):
        return "Dedalus state"


class StateDedalus(StateBase):
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        info_solver.classes.State._set_attribs(
            {
                "keys_state_phys": ["b", "vx", "vz"],
                "keys_computable": ["dz_b", "dz_vx", "dz_vz"],
                "keys_phys_needed": ["b", "vx", "vz"],
            }
        )

    def compute(self, key):
        """Compute scalar fields such a component of the velocity or vorticity."""
        return self.state_phys.get_var(key)

    def __init__(self, sim, oper=None):
        super().__init__(sim, oper)
        sim.init_dedalus()
        if self.sim.params.ONLY_COARSE_OPER:
            solver = None
        else:
            solver = self.sim.dedalus_solver

        self.state_phys = StatePhysDedalus(solver, self.keys_state_phys)
