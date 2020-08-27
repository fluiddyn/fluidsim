import unittest

from fluidsim.base.params import merge_params

from fluidsim.util.testing import TestCase


class TestParameters(TestCase):
    """Test Parameters class and related functions."""

    def test_merge(self):
        """Test merging parameters."""
        from fluidsim.base.solvers.base import SimulBase
        from fluidsim.solvers.ad1d.solver import Simul

        params1 = SimulBase.create_default_params()
        params2 = Simul.create_default_params()
        merge_params(params1, params2)


if __name__ == "__main__":
    unittest.main()
