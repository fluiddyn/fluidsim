"""Output (:mod:`fluidsim.solvers.models0d.predaprey.output`)
=============================================================

Provides the modules:

.. autosummary::
   :toctree:

   print_stdout

and the main output class for this solver:

.. autoclass:: Output
   :members:
   :private-members:

"""

from math import log

from fluidsim.base.output import OutputBase


class Output(OutputBase):
    """Output for prepaprey solver."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""

        OutputBase._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes
        package = "fluidsim.solvers.models0d.predaprey.output"

        classes.PrintStdOut.module_name = package + ".print_stdout"
        classes.PrintStdOut.class_name = "PrintStdOutPredaPrey"

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """Complete the `params` container (static method)."""
        OutputBase._complete_params_with_default(params, info_solver)

        params.output.phys_fields.field_to_plot = "X"

    def compute_potential(self):
        """Compute energy(k)"""

        p = self.sim.params
        X = self.sim.state.state_phys.get_var("X")
        Y = self.sim.state.state_phys.get_var("Y")
        return p.C * log(X) - p.D * X + p.A * log(Y) - p.B * Y
