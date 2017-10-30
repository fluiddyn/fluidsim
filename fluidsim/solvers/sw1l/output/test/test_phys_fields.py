from __future__ import print_function

import unittest

from fluidsim.solvers.test.test_ns import run_mini_simul
from . import BaseTestCase, mpi


class TestPhysFields(BaseTestCase):
    _tag = 'phys_fields'

    @classmethod
    def setUpClass(cls):
        cls.sim = run_mini_simul(cls.solver, HAS_TO_SAVE=True)
        cls.output = cls.sim.output
        cls.module = getattr(cls.output, cls._tag)

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_phys_fields(self):
        self._plot()


if __name__ == '__main__':
    unittest.main()
