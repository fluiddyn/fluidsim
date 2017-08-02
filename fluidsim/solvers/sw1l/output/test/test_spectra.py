from __future__ import print_function

import unittest

from . import BaseTestCase


class TestSpectra(BaseTestCase):
    _tag = 'spectra'

    def test_plot_spectra(self):
        self.module.plot = self.module.plot2d
        self._plot()
        self._online_plot(*self.dico)


if __name__ == '__main__':
    unittest.main()
