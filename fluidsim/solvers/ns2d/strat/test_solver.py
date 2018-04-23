# -*- coding: utf-8 -*-
from __future__ import division

import unittest
import shutil

import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected

from fluidsim.solvers.ns2d.strat.solver import Simul


class TestSolverNS2DStrat(unittest.TestCase):
    Simul = Simul

    def tearDown(self):
        # clean by removing the directory
        if mpi.rank == 0:
            if hasattr(self, "sim"):
                shutil.rmtree(self.sim.output.path_run)

    def test_tendency(self):

        params = self.Simul.create_default_params()

        params.short_name_type_run = "test"

        nh = 32
        params.oper.nx = nh
        params.oper.ny = nh
        Lh = 6.
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.oper.coef_dealiasing = 2. / 3
        params.nu_8 = 2.

        params.time_stepping.t_end = 0.5

        params.init_fields.type = "noise"
        params.output.HAS_TO_SAVE = False

        with stdout_redirected():
            self.sim = sim = self.Simul(params)

        rot_fft = sim.state.get_var("rot_fft")
        b_fft = sim.state.get_var("b_fft")

        tend = sim.tendencies_nonlin(state_spect=sim.state.state_spect)
        f_rot_fft = tend.get_var("rot_fft")
        f_b_fft = tend.get_var("b_fft")

        sim.check_energy_conservation(rot_fft, b_fft, f_rot_fft, f_b_fft)

    def test_forcing_output(self):

        params = self.Simul.create_default_params()

        params.short_name_type_run = "test"

        nh = 48
        params.oper.nx = 2 * nh
        params.oper.ny = nh
        lh = 6.
        params.oper.Lx = lh
        params.oper.Ly = lh

        params.oper.coef_dealiasing = 2. / 3
        params.nu_8 = 2.

        params.time_stepping.t_end = 0.5

        params.init_fields.type = "noise"
        params.forcing.enable = True
        params.forcing.nkmax_forcing = 20
        params.forcing.nkmin_forcing = 2
        params.forcing.type = "tcrandom_anisotropic"
        params.forcing.tcrandom.time_correlation = 0.2
        params.forcing.tcrandom_anisotropic.angle = u"30°"

        # save all outputs!
        periods = params.output.periods_save
        for key in periods._key_attribs:
            periods[key] = 0.2

        with stdout_redirected():
            self.sim = sim = self.Simul(params)
            sim.time_stepping.start()

            if mpi.nb_proc == 1:
                sim.output.spectra.plot1d()
                sim.output.spectra.plot2d()


# sim.output.spatial_means.plot()
# sim.output.print_stdout.plot_energy()
# sim.output.print_stdout.plot_deltat()

# sim.output.spect_energy_budg.plot()
# with self.assertRaises(ValueError):
#     sim.state.get_var('test')

# sim2 = fls.load_sim_for_plot(sim.output.path_run)
# sim2.output

# sim2.output.increments.load()
# sim2.output.increments.plot()
# sim2.output.increments.load_pdf_from_file()


# class TestSolverNS2DFluidfft(TestSolverNS2D):
#     Simul = Simul2


if __name__ == "__main__":
    unittest.main()
