from __future__ import division

import unittest
import shutil

import numpy as np


import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected

from fluidsim.solvers.ad1d.solver import Simul


class TestSolverAD1D(unittest.TestCase):
    def test_simul(self):

        params = Simul.create_default_params()

        params.U = 1.

        params.short_name_type_run = 'test'

        params.oper.nx = 40
        params.oper.Lx = 1.

        params.time_stepping.type_time_scheme = 'RK2'

        params.nu_2 = 0.01

        params.time_stepping.t_end = 0.4
        params.time_stepping.USE_CFL = True

        params.init_fields.type = 'gaussian'

        params.output.periods_print.print_stdout = 0.25

        params.output.periods_save.phys_fields = 0.5

        params.output.periods_plot.phys_fields = 0.

        params.output.phys_fields.field_to_plot = 's'

        with stdout_redirected():
            sim = Simul(params)

        # clean by removing the directory
        if mpi.rank == 0:
            shutil.rmtree(sim.output.path_run)


if __name__ == '__main__':
    unittest.main()
