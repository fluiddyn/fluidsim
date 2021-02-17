"""Mechanism to extend a Simul class with just a simple class
=============================================================

"""


class SimulExtender:
    @classmethod
    def create_extended_Simul(cls, Simul):
        """Should return the new extended Simul class

        """
        return Simul

    @classmethod
    def _complete_params_with_default(cls, params):
        cls.complete_params_with_default(params)

    @classmethod
    def complete_params_with_default(cls, params):
        """Should complete the simul parameters"""
        pass

    @classmethod
    def add_info_solver_modificator(self, InfoSolver, modif_info_solver):
        if not hasattr(InfoSolver, "_modificators"):
            InfoSolver._modificators = []
        InfoSolver._modificators.append(modif_info_solver)
