from .solver import Simul

from fluidsim.solvers.ad1d.test_solver import TestSolverAD1D as Base


class TestSolverAD1DPseudoSpectral(Base):
    Simul = Simul
