"""Mechanism to extend a Simul class with just a simple class
=============================================================

.. autoclass:: SimulExtender
   :members:
   :private-members:

.. autosummary::
   :toctree:

   spatial_means_regions_milestone

"""


class SimulExtender:
    """Abstract class to define a "Simul extender"

    Simul extenders are classes that can extend a ``Simul`` class to change its
    behavior for some simulations.

    This class is meant to be subclassed. The child class has to contain one
    class attribute ``_module_name`` and two class methods
    ``create_extended_Simul`` and ``complete_params_with_default``. An example
    can be found in the module
    :mod:`fluidsim.extend_simul.spatial_means_regions_milestone`.
    """

    @classmethod
    def add_info_solver_modificator(cls, InfoSolver, modif_info_solver):
        if not hasattr(InfoSolver, "_modificators"):
            InfoSolver._modificators = []
            InfoSolver._extenders = []
        InfoSolver._modificators.append(modif_info_solver)
        InfoSolver._extenders.append((cls._module_name, cls.__name__))

    @classmethod
    def create_extended_Simul(cls, Simul, modif_info_solver=None):
        """Should return the new extended Simul class"""
        if modif_info_solver is None:
            return Simul

        class NewInfoSolver(Simul.InfoSolver):
            pass

        cls.add_info_solver_modificator(NewInfoSolver, modif_info_solver)

        class NewSimul(Simul):
            InfoSolver = NewInfoSolver

        return NewSimul

    @classmethod
    def _complete_params_with_default(cls, params):
        cls.complete_params_with_default(params)

    @classmethod
    def complete_params_with_default(cls, params):
        """Should complete the simul parameters"""
