from __future__ import print_function

import unittest

from . import BaseTestCase


class TestSpatialMeans(BaseTestCase):
    _tag = 'spatial_means'

    def test_plot_spatial_means(self):
        self._plot()


if __name__ == '__main__':
    unittest.main()
