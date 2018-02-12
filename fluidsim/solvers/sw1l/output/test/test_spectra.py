from __future__ import print_function

import unittest

from . import BaseTestCase, mpi


class TestSpectra(BaseTestCase):
    _tag = 'spectra'

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_spectra(self):
        self.module.plot = self.module.plot1d
        self._plot()

    def test_online_plot_spectra(self):
        self._online_plot_saving(*self.dico)


class TestExactlin(TestSpectra):
    solver = 'sw1l.exactlin'

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_spectra(self):
        self.module.plot = self.module.plot2d
        self._plot()



if __name__ == '__main__':
    unittest.main()
