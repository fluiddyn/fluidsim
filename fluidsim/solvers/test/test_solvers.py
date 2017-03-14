from __future__ import division

import unittest
from shutil import rmtree

import fluidsim
import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected


def run_mini_simul(key_solver, HAS_TO_SAVE=False, FORCING=False):

    Simul = fluidsim.import_simul_class_from_key(key_solver)

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    nh = 16
    params.oper.nx = nh
    params.oper.ny = nh
    Lh = 6.
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    params.oper.coef_dealiasing = 2./3
    params.nu_8 = 2.

    try:
        params.f = 1.
        params.c2 = 200.
    except AttributeError:
        pass

    params.time_stepping.t_end = 0.5

    params.init_fields.type = 'dipole'

    if HAS_TO_SAVE:
        params.output.periods_save.spectra = 0.5
        params.output.periods_save.spatial_means = 0.5
        params.output.periods_save.spect_energy_budg = 0.5

    if FORCING:
        params.FORCING = True
        params.forcing.type = 'waves'

    params.output.HAS_TO_SAVE = HAS_TO_SAVE

    with stdout_redirected():
        sim = Simul(params)
        sim.time_stepping.start()

    if HAS_TO_SAVE:
        sim.output.spatial_means.load()

    return sim


class TestSolver(unittest.TestCase):
    solver = 'NS2D'
    options = {'HAS_TO_SAVE': False, 'FORCING': False}

    def setUp(self):
        self.sim = run_mini_simul(self.solver, **self.options)

    def tearDown(self):
        if mpi.rank == 0:
            rmtree(self.sim.output.path_run)

    def test(self):
        pass


class TestSW1L(TestSolver):
    solver = 'SW1L'
    options = {'HAS_TO_SAVE': True, 'FORCING': False}


class TestSW1LOnlyWaves(TestSW1L):
    solver = 'SW1L.onlywaves'
    options = {'HAS_TO_SAVE': True, 'FORCING': False}


class TestSW1LExactLin(TestSW1L):
    solver = 'SW1L.exactlin'
    options = {'HAS_TO_SAVE': True, 'FORCING': False}


class TestSW1LModify(TestSW1L):
    solver = 'SW1L.modified'
    options = {'HAS_TO_SAVE': True, 'FORCING': False}

if __name__ == '__main__':
    unittest.main()
