from __future__ import division

import unittest
import shutil

import numpy as np

import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected

from fluidsim.solvers.plate2d.solver import Simul


class TestSolverPlate2D(unittest.TestCase):
    def tearDown(self):
        if mpi.rank == 0:
            shutil.rmtree(self.sim.output.path_run)

    def test_tendency(self):

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

        params.time_stepping.USE_CFL = False
        params.time_stepping.deltat0 = 0.005
        params.time_stepping.t_end = 0.5

        params.init_fields.type = 'noise'
        params.output.HAS_TO_SAVE = False
        params.forcing.enable = False

        params.output.ONLINE_PLOT_OK = False

        with stdout_redirected():
            self.sim = sim = Simul(params)

        ratio = sim.test_tendencies_nonlin()

        self.assertGreater(2e-15, ratio)

    def test_output(self):

        params = Simul.create_default_params()

        params.short_name_type_run = 'test'

        nh = 24
        Lh = 1.
        params.oper.nx = nh
        params.oper.ny = nh
        params.oper.Lx = Lh
        params.oper.Ly = Lh
        params.oper.coef_dealiasing = 2. / 3

        delta_x = Lh / nh

        kmax = np.sqrt(2)*np.pi/delta_x
        deltat = 2*np.pi/kmax**2/2

        params.time_stepping.USE_CFL = False
        params.time_stepping.deltat0 = deltat
        params.time_stepping.USE_T_END = False
        params.time_stepping.it_end = 16

        params.init_fields.type = 'noise'
        params.init_fields.noise.velo_max = 1e-6

        params.forcing.enable = True
        params.forcing.type = 'tcrandom'
        params.forcing.forcing_rate = 1e4
        params.forcing.nkmax_forcing = 5
        params.forcing.nkmin_forcing = 2
        params.forcing.tcrandom.time_correlation = 100*deltat

        params.nu_8 = 2e1*params.forcing.forcing_rate**(1./3)*delta_x**8

        params.output.periods_print.print_stdout = 0.05

        params.output.periods_save.phys_fields = 5.
        params.output.periods_save.spectra = 0.05
        params.output.periods_save.spatial_means = 10*deltat
        params.output.periods_save.correl_freq = 1

        params.output.ONLINE_PLOT_OK = False
        params.output.period_refresh_plots = 0.05

        params.output.correl_freq.HAS_TO_PLOT_SAVED = False
        params.output.correl_freq.it_start = 0
        params.output.correl_freq.nb_times_compute = 8
        params.output.correl_freq.coef_decimate = 1
        params.output.correl_freq.iomegas1 = [1, 2]

        with stdout_redirected():
            self.sim = sim = Simul(params)
            sim.time_stepping.start()

        sim.output.correl_freq.compute_corr4_norm()


if __name__ == '__main__':
    unittest.main()
