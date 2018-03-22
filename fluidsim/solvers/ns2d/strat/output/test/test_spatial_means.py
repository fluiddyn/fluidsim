from __future__ import print_function

import unittest
import numpy as np

from fluidsim.solvers.ns2d.strat.output.test import TestNS2DStrat, mpi


class TestSpatialMeans(TestNS2DStrat):
    _tag = 'spatial_means'

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_spatial_means(self):
        self._plot()


if __name__ == '__main__':
    unittest.main()
