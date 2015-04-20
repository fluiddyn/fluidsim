
import unittest
import shutil

import fluiddyn as fld
import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected


def run_mini_simul(key_solver):

    solver = fld.simul.import_module_solver_from_key(key_solver)

    params = solver.Simul.create_default_params()

    params.short_name_type_run = 'test'

    nh = 64
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

    params.output.HAS_TO_SAVE = False

    with stdout_redirected():
        sim = solver.Simul(params)
        sim.time_stepping.start()

    # clean by removing the directory
    if mpi.rank == 0:
        shutil.rmtree(sim.output.path_run)


class TestSolvers(unittest.TestCase):
    def test_ns2d(self):
        """Should be able to run a NS2D simul."""
        run_mini_simul('NS2D')

    def test_sw1l(self):
        """Should be able to run a SW1L simul."""
        run_mini_simul('SW1L')

    def test_sw1l_onlywaves(self):
        """Should be able to run a SW1L.onlywaves simul."""
        run_mini_simul('SW1L.onlywaves')

    def test_sw1l_exactlin(self):
        """Should be able to run a SW1L.exactlin simul."""
        run_mini_simul('SW1L.exactlin')


if __name__ == '__main__':
    unittest.main()
