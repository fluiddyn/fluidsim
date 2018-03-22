from __future__ import print_function

import unittest
import numpy as np

from fluidsim.solvers.ns2d.strat.output.test import TestNS2DStrat, mpi


class TestSpatialMeans(TestNS2DStrat):
    _tag = 'spatial_means'

    def test_energy_conservation(self):
        """Check conservation of energy (no viscosity & no dissipation)"""
        dico = self.dico
        energies = dico['E']

        # Compute energy difference
        dE = []
        for i in range(len(energies)-1):
            difference = abs(energies[i+1] - energies[i])
            dE.append(difference)

        self.assertAlmostEqual(np.mean(dE), 0)

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_spatial_means(self):
        self._plot()


if __name__ == '__main__':
    unittest.main()
