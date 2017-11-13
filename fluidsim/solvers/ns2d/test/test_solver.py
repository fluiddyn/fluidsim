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

        rot_fft = sim.state('rot_fft')

        tend = sim.tendencies_nonlin(state_fft=sim.state.state_fft)
        Frot_fft = tend.get_var('rot_fft')

        T_rot = np.real(Frot_fft.conj()*rot_fft)

        ratio = (sim.oper.sum_wavenumbers(T_rot) /
                 sim.oper.sum_wavenumbers(abs(T_rot)))

        self.assertGreater(1e-15, ratio)

        # print ('sum(T_rot) = {0:9.4e} ; '
        #        'sum(abs(T_rot)) = {1:9.4e}').format(
        #            sim.oper.sum_wavenumbers(T_rot),
        #            sim.oper.sum_wavenumbers(abs(T_rot)))

    def test_forcing(self):

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
        params.FORCING = True
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
                    sim.state.compute('test')

                sim2 = fls.load_sim_for_plot(sim.output.path_run)
                sim2.output

            # `compute('q')` two times for better coverage...
            sim.state.compute('q')
            sim.state.compute('q')
            sim.state.compute('div')

            path_run = sim.output.path_run
            if mpi.nb_proc > 1:
                path_run = mpi.comm.bcast(path_run)

            sim3 = fls.load_state_phys_file(path_run)
            sim3.params.time_stepping.t_end += 1.
            sim3.time_stepping.start()


# class TestSolverNS2DFluidfft(TestSolverNS2D):
#     Simul = Simul2


if __name__ == '__main__':
    unittest.main()
