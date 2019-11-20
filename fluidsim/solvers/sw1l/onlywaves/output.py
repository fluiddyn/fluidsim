""" """

import numpy as np

from fluidsim.solvers.sw1l.output import OutputSW1L


class OutputSW1LWaves(OutputSW1L):
    """subclass :class:`OutputSW1L`"""

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        OutputSW1L._complete_info_solver(info_solver)

        classes = info_solver.classes.Output.classes

        classes.SpectralEnergyBudget.class_name = (
            "SpectralEnergyBudgetSW1L"  # Waves'
        )
