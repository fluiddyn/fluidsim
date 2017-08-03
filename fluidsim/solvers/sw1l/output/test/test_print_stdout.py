from __future__ import print_function

import unittest

import numpy as np

from . import BaseTestCase, mpi


class TestPrintStdout(BaseTestCase):
    _tag = 'print_stdout'

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_print_stdout(self):
        self._plot()

    def test_energy_vs_spatial_means(self):
        '''Verify energy saved by spatial_means module is the same.'''
        dico_spatial_means = self.output.spatial_means.load()
        try:
            self.assertTrue(
                np.allclose(self.dico['E'], dico_spatial_means['E'], atol=1.e-4))
        except AssertionError:
            print(self.dico['E'], dico_spatial_means['E'])
            raise


if __name__ == '__main__':
    unittest.main()
