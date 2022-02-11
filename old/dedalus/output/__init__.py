"""Module to generate output from Dedalus (:mod:`fluidsim.base.dedalus.output`)
==================================================================================

Provides:

.. autoclass:: OutputDedalus
   :members:
   :private-members:

"""

from ...output.base import OutputBase  # , SpecificOutput


class OutputDedalus(OutputBase):
    @staticmethod
    def _complete_info_solver(info_solver):

        OutputBase._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        base_name_mod = "fluidsim.solvers.ns2d.output"

        classes.PrintStdOut.module_name = base_name_mod + ".print_stdout"
        classes.PrintStdOut.class_name = "PrintStdOutNS2D"

        classes.PhysFields.class_name = "PhysFieldsBase2D"

    def compute_energy(self):
        """Compute the spatially averaged energy."""
        vx = self.sim.state.get_var("vx")
        vz = self.sim.state.get_var("vz")
        return 0.5 * (vx**2 + vz**2).mean()
