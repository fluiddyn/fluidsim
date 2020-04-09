from .solver import Simul

from ..test_solver import TestSolverAD1D as Base


class TestSolverAD1DPseudoSpectral(Base):
    Simul = Simul
