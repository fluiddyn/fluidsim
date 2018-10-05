from fluidsim.base.output import OutputBasePseudoSpectral


class Output(OutputBasePseudoSpectral):
    """Output for spherical harmo solvers."""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the `info_solver` container (static method)."""

        OutputBasePseudoSpectral._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        classes.PhysFields.module_name = (
            "fluidsim.base.sphericalharmo.phys_fields"
        )
        classes.PhysFields.class_name = "PhysFieldsSphericalHarmo"
