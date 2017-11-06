import unittest

from fluiddyn.io import stdout_redirected
from fluidsim.base.params import merge_params
from fluidsim.base.solvers.base import SimulBase
from fluidsim.base.solvers.pseudo_spect import SimulBasePseudoSpectral as Simul


class TestParameters(unittest.TestCase):
    """Test Parameters class and related functions."""

    def test_merge(self):
        """Test merging parameters."""
        params1 = SimulBase.create_default_params()
        params2 = Simul.create_default_params()
        with stdout_redirected():
            merge_params(params1, params2)


if __name__ == '__main__':
    unittest.main()
