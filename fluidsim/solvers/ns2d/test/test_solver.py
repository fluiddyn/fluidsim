from __future__ import division

import unittest
import shutil

import numpy as np

import fluidsim as fls

import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected

from fluidsim.solvers.ns2d.solver import Simul
# from fluidsim.solvers.ns2d.solver_oper_cython import Simul as Simul2


class TestSolverNS2D(unittest.TestCase):
    Simul = Simul

    def tearDown(self):
        # clean by removing the directory
        if mpi.rank == 0:
            if hasattr(self, 'sim'):
                shutil.rmtree(self.sim.output.path_run)

    def test_tendency(self):

        params = self.Simul.create_default_params()

        params.short_name_type_run = 'test'

        nh = 32
        params.oper.nx = nh
        params.oper.ny = nh
        Lh = 6.
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.oper.coef_dealiasing = 2./3
        params.nu_8 = 2.

        params.time_stepping.t_end = 0.5

        params.init_fields.type = 'noise'
        params.output.HAS_TO_SAVE = False

        with stdout_redirected():
            self.sim = sim = self.Simul(params)

        rot_fft = sim.state.get_var('rot_fft')

        tend = sim.tendencies_nonlin(state_spect=sim.state.state_spect)
        Frot_fft = tend.get_var('rot_fft')

        T_rot = np.real(Frot_fft.conj()*rot_fft)

        ratio = (sim.oper.sum_wavenumbers(T_rot) /
                 sim.oper.sum_wavenumbers(abs(T_rot)))

        self.assertGreater(1e-15, ratio)

        # print ('sum(T_rot) = {0:9.4e} ; '
        #        'sum(abs(T_rot)) = {1:9.4e}').format(
        #            sim.oper.sum_wavenumbers(T_rot),
        #            sim.oper.sum_wavenumbers(abs(T_rot)))

    def test_forcing_output(self):

        params = self.Simul.create_default_params()

        params.short_name_type_run = 'test'

        nh = 16
        params.oper.nx = 2*nh
        params.oper.ny = nh
        Lh = 6.
        params.oper.Lx = Lh
        params.oper.Ly = Lh

        params.oper.coef_dealiasing = 2./3
        params.nu_8 = 2.

        params.time_stepping.t_end = 0.5

        params.init_fields.type = 'noise'
        params.forcing.enable = True
        params.forcing.type = 'tcrandom'

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

                sim.output.spatial_means.plot()
                sim.output.print_stdout.plot_energy()
                sim.output.print_stdout.plot_deltat()

                sim.output.spect_energy_budg.plot()
                with self.assertRaises(ValueError):
                    sim.state.get_var('test')

                sim2 = fls.load_sim_for_plot(sim.output.path_run)
                sim2.output

                sim2.output.increments.load()
                sim2.output.increments.plot()
                sim2.output.increments.load_pdf_from_file()

            # `compute('q')` two times for better coverage...
            sim.state.get_var('q')
            sim.state.get_var('q')
            sim.state.get_var('div')

            path_run = sim.output.path_run
            if mpi.nb_proc > 1:
                path_run = mpi.comm.bcast(path_run)

            sim3 = fls.load_state_phys_file(path_run, modif_save_params=False)
            sim3.params.time_stepping.t_end += 0.2
            sim3.time_stepping.start()

        if mpi.rank == 0:
            sim3.output.phys_fields.animate(
                'ux', dt_frame_in_sec=1e-6, dt_equations=0.3, repeat=False,
                clim=(-1, 1), save_file=False, numfig=1)



# class TestSolverNS2DFluidfft(TestSolverNS2D):
#     Simul = Simul2


if __name__ == '__main__':
    unittest.main()
