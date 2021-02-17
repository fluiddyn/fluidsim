from .solver import Simul

from fluidsim.solvers.ad1d.test_solver import TestSolverAD1D
from fluidsim.util.testing import skip_if_no_fluidfft


@skip_if_no_fluidfft
class TestSolverAD1DPseudoSpectral(TestSolverAD1D):
    Simul = Simul


del TestSolverAD1D
