
import unittest
import shutil
import numpy as np

import fluidsim
import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected


def run_mini_simul(key_solver, HAS_TO_SAVE=False, FORCING=False):

    Simul = fluidsim.import_simul_class_from_key(key_solver)

    params = Simul.create_default_params()

    params.short_name_type_run = 'test'

    nh = 32
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
        params.output.periods_save.spect_energy_budg = 0.5

    if FORCING:
        params.FORCING = True
        params.forcing.type = 'waves'

    params.output.HAS_TO_SAVE = HAS_TO_SAVE

    with stdout_redirected():
        sim = Simul(params)
        sim.time_stepping.start()

    return sim

def clean_simul(sim):
    # clean by removing the directory
    if mpi.rank == 0:
        shutil.rmtree(sim.output.path_run)



class TestSolvers(unittest.TestCase):
    def test_ns2d(self):
        """Should be able to run a NS2D simul."""
        self.sim = run_mini_simul('NS2D')
        clean_simul(self.sim)

    def test_sw1l(self):
        """Should be able to run a SW1L simul."""
        self.sim = run_mini_simul('SW1L', HAS_TO_SAVE=True, FORCING=True)
        clean_simul(self.sim)

    def test_sw1l_onlywaves(self):
        """Should be able to run a SW1L.onlywaves simul."""
        self.sim = run_mini_simul('SW1L.onlywaves')
        clean_simul(self.sim)

    def test_sw1l_exactlin(self):
        """Should be able to run a SW1L.exactlin simul."""
        self.sim = run_mini_simul('SW1L.exactlin')
        clean_simul(self.sim)

if __name__ == '__main__':
    unittest.main()
