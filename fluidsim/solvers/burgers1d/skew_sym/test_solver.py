from fluidsim.util.testing import skip_if_no_fluidfft

from fluidsim.solvers.burgers1d.test_solver import TestSolverSquare1D as BaseTest
from fluidsim.solvers.burgers1d.skew_sym.solver import Simul


@skip_if_no_fluidfft
class TestSolverSkewSym1D(BaseTest):
    Simul = Simul


del BaseTest
