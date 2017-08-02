from __future__ import print_function

import unittest
from shutil import rmtree

import fluiddyn.util.mpi as mpi
from fluidsim.solvers.test.test_solvers import run_mini_simul

from .test_spect_energy_budg import TestSpectEnergyBudg


class TestOutputSW1L(TestSpectEnergyBudg, unittest.TestCase):
    solver = 'sw1l'

    @classmethod
    def setUpClass(cls):
        cls.sim = run_mini_simul(cls.solver, HAS_TO_SAVE=True)
        cls.output = cls.sim.output
        super(TestOutputSW1L, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        if mpi.rank == 0:
            rmtree(cls.sim.output.path_run)


class TestOutputWaves(TestOutputSW1L):
    solver = 'sw1l.onlywaves'


class TestOutputExactlin(TestOutputSW1L):
    solver = 'sw1l.exactlin'


class TestOutputExmod(TestOutputSW1L):
    solver = 'sw1l.exactlin.modified'


class TestOutputModif(TestOutputSW1L):
    solver = 'sw1l.modified'


if __name__ == '__main__':
    unittest.main()
