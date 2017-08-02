from __future__ import print_function

import unittest

from . import BaseTestCase, mpi


class TestSpectra(BaseTestCase):
    _tag = 'spectra'

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_spectra(self):
        self.module.plot = self.module.plot2d
        self._plot()

    def test_online_plot_spectra(self):
        self._online_plot(*self.dico)


if __name__ == '__main__':
    unittest.main()
