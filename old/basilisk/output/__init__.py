"""Module to generate output from Basilisk (:mod:`fluidsim.base.basilisk.output`)
==================================================================================

Provides:

.. autoclass:: OutputBasilisk
   :members:
   :private-members:

"""

from ...output.base import OutputBase  # , SpecificOutput


class OutputBasilisk(OutputBase):
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
        ux = self.sim.state.get_var("ux")
        uy = self.sim.state.get_var("uy")
        return 0.5 * (ux**2 + uy**2).mean()

    def compute_enstrophy(self):
        """Compute the spatially averaged enstrophy."""
        rot = self.sim.state.get_var("rot")
        return 0.5 * (rot**2).mean()
