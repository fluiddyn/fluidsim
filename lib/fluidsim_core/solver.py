"""The Simulation class API.

.. autoclass:: SimulCore
   :members: _complete_params_with_default, create_default_params, __init__

"""
from abc import ABC, abstractclassmethod, abstractmethod, abstractstaticmethod
from warnings import warn

from .info import InfoSolverCore, create_info_simul
from .params import Parameters


__all__ = ["SimulCore"]


class SimulCore(ABC):
    """Represent a solver. This is an abstract base class describing the
    bare minimum number of methods to have a working solver.

    """

    InfoSolver = InfoSolverCore
    Parameters = Parameters

    @abstractstaticmethod
    def _complete_params_with_default(params):
        """A static method used to complete the *params* container."""
        attribs = {}
        params._set_attribs(attribs)
        #  params._set_doc("""Describe docstrings for params here.""")

    @abstractclassmethod
    def create_default_params(cls):
        """Sets an ``info_solver`` instance as a class attribute and returns a
        *params* container populated with default values.

        """
        cls.info_solver = cls.InfoSolver()
        return cls.Parameters._create_params(cls.info_solver)

    @abstractmethod
    def __init__(self, params):
        """Instantiate Simulation class and optionally all classes as specified
        by :any:`fluidsim_core.info.InfoSolverCore.import_classes`"""
        if not hasattr(self, "info_solver") or not isinstance(
            self.info_solver, self.InfoSolver
        ):
            """The condition `not isinstance(self.info_solver, self.InfoSolver)`
            is needed in some rare situations (mostly testing) for which
            info_solver comes from an inherited class."""
            self.info_solver = self.InfoSolver()

        if not isinstance(params, self.Parameters):
            raise TypeError(
                f"params should be a {self.Parameters} instance, "
                f"not {type(params)}"
            )

        self.params = params
        self.info = create_info_simul(self.info_solver, params)

        # Instantiate other classes as follows
        # ------------------------------------
        # dict_classes = self.info_solver.import_classes()
        # Operators = dict_classes["Operators"]
        # self.oper = Operators(params=params)
        # ..
