from __future__ import print_function

import unittest

from fluidsim.solvers.test.test_ns import run_mini_simul
from . import BaseTestCase, mpi


class TestSpatialMeans(BaseTestCase):
    _tag = 'spatial_means'

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_spatial_means(self):
        self._plot()


class TestSpatialMeansForced(TestSpatialMeans):
    solver = 'sw1l.exactlin'

    @classmethod
    def setUpClass(cls):
        cls.sim = run_mini_simul(cls.solver, HAS_TO_SAVE=True,
                                 forcing_enable=True)
        cls.output = cls.sim.output
        cls.module = module = getattr(cls.output, cls._tag)
        cls.dico = module.load()


if __name__ == '__main__':
    unittest.main()
