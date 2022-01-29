import numpy as np

from fluidsim.base.output import OutputBase


class Output(OutputBase):
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        classes = info_solver.classes.Output._set_child("classes")

        base_name_mod = "fluidsim.solvers.ad1d.output"

        classes._set_child(
            "PrintStdOut",
            attribs={
                "module_name": base_name_mod + ".print_stdout",
                "class_name": "PrintStdOutAD1D",
            },
        )

        classes._set_child(
            "PhysFields",
            attribs={
                "module_name": "fluidsim.base.output.phys_fields1d",
                "class_name": "PhysFieldsBase1D",
            },
        )

    @staticmethod
    def _complete_params_with_default(params, info_solver):
        """This static method is used to complete the *params* container."""
        OutputBase._complete_params_with_default(params, info_solver)

        params.output.phys_fields.field_to_plot = "s"

    def compute_energy(self):
        return 0.5 * np.mean(self.sim.state.state_phys**2)
