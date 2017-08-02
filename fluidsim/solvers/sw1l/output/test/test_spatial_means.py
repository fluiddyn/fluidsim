from __future__ import print_function

import unittest

from . import BaseTestCase, mpi


class TestSpatialMeans(BaseTestCase):
    _tag = 'spatial_means'

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_spatial_means(self):
        self._plot()


if __name__ == '__main__':
    unittest.main()
