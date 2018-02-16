from __future__ import division

import unittest
from shutil import rmtree

import numpy as np

import fluidsim
from fluiddyn.util import mpi
from fluiddyn.io import stdout_redirected


def run_mini_simul(
        key_solver, nh=16, init_fields='dipole', name_run='test',
        type_forcing='waves', HAS_TO_SAVE=False, forcing_enable=False):

    Simul = fluidsim.import_simul_class_from_key(key_solver)

    params = Simul.create_default_params()

    params.short_name_type_run = name_run

    params.oper.nx = nh
    params.oper.ny = nh
    Lh = 6.
    params.oper.Lx = Lh
    params.oper.Ly = Lh

    params.oper.coef_dealiasing = 2. / 3
    params.nu_8 = 2.

    try:
        params.f = 1.
        params.c2 = 200.
    except AttributeError:
        pass

    params.time_stepping.t_end = 0.5

    params.init_fields.type = init_fields

    if HAS_TO_SAVE:
        params.output.periods_save.spectra = 0.25
        params.output.periods_save.spatial_means = 0.25
        params.output.periods_save.spect_energy_budg = 0.25

    if forcing_enable:
        params.forcing.enable = True
        params.forcing.type = type_forcing
        params.forcing.nkmin_forcing = 2
        params.forcing.nkmax_forcing = 4

    params.output.HAS_TO_SAVE = HAS_TO_SAVE

    with stdout_redirected():
        sim = Simul(params)
        sim.time_stepping.start()

    if HAS_TO_SAVE:
        sim.output.spatial_means.load()

    return sim


class TestSolver(unittest.TestCase):
    solver = 'NS2D'
    options = {'HAS_TO_SAVE': False, 'forcing_enable': False}
    zero = 1e-15

    @classmethod
    def setUpClass(cls):
        cls.sim = run_mini_simul(cls.solver, **cls.options)

    @classmethod
    def tearDownClass(cls):
        path_run = cls.sim.output.path_run
        del cls.sim
        if mpi.rank == 0:
            rmtree(path_run)

    def assertZero(self, value, places=None, msg=None, warn=True):
        if places is None:
            places = -int(np.log10(self.zero))

        self.assertAlmostEqual(value, 0, places=places, msg=msg)
        if places < 7 and mpi.rank == 0 and warn:
            import warnings
            warnings.warn(
                'Machine zero level too high. Value to be asserted as zero' +
                '= {} : {}'.format(value, self.id()))

    def test_enstrophy_conservation(self, zero_places=None):
        """Verify that the enstrophy growth rate due to nonlinear tendencies
        (advection term) must be zero.

        """
        self.sim.params.forcing.enable = False
        tendencies_fft = self.sim.tendencies_nonlin()
        state_spect = self.sim.state.state_spect
        oper = self.sim.oper
        Frot_fft = tendencies_fft.get_var('rot_fft')
        rot_fft = state_spect.get_var('rot_fft')

        T_rot = (Frot_fft.conj() * rot_fft +
                 Frot_fft * rot_fft.conj()).real / 2.
        sum_T = oper.sum_wavenumbers(T_rot)
        self.assertZero(sum_T, zero_places)


class TestNS2DStrat(TestSolver):
    solver = 'NS2D.strat'
    options = {'HAS_TO_SAVE': False, 'forcing_enable': False}

    def test_enstrophy_conservation(self):
        # This solver does not solve for vertical component of vorticity
        pass

    def test_energy_conservation(self):
        """Verify that the energy growth rate due to nonlinear tendencies
        (advection term) must be zero.

        """
        self.sim.params.forcing.enable = False
        tendencies_fft = self.sim.tendencies_nonlin()

        oper = self.sim.oper
        state_spect = self.sim.state.state_spect
        Frot_fft = tendencies_fft.get_var('rot_fft')
        rot_fft = state_spect.get_var('rot_fft')
        Fx_fft, Fy_fft = oper.vecfft_from_rotfft(Frot_fft)
        Fb_fft = tendencies_fft.get_var('b_fft')
        ux_fft, uy_fft = oper.vecfft_from_rotfft(rot_fft)
        b_fft = state_spect.get_var('b_fft')

        transferEK = np.real(
            ux_fft.conj() * Fx_fft + uy_fft.conj() * Fy_fft)
        transferEA = (1. / self.sim.params.N ** 2) * np.real(
            b_fft.conj() * Fb_fft)

        T_tot = transferEK + transferEA
        sum_T = oper.sum_wavenumbers(T_tot)
        self.assertZero(sum_T)


if __name__ == '__main__':
    unittest.main()
