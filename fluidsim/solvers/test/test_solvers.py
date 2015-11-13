
import unittest
import shutil
import numpy as np

import fluidsim
import fluiddyn.util.mpi as mpi
from fluiddyn.io import stdout_redirected
from fluidsim.base.output.spect_energy_budget import inner_prod


def run_mini_simul(key_solver, HAS_TO_SAVE=False):

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

    params.output.HAS_TO_SAVE = HAS_TO_SAVE

    with stdout_redirected():
        sim = Simul(params)
        sim.time_stepping.start()

    return sim

def clean_simul(sim):
    # clean by removing the directory
    if mpi.rank == 0:
        shutil.rmtree(sim.output.path_run)


class TestOutput(object):
    def verify_sw1l_spect_energy_budg(self):
        module = self.sim.output.spect_energy_budg
        ux_fft = self.sim.state('ux_fft')
        uy_fft = self.sim.state('uy_fft')
        eta_fft = self.sim.state('eta_fft')
        ux = self.sim.state.state_phys.get_var('ux')
        uy = self.sim.state.state_phys.get_var('uy')
        dico_results = module.compute()
        b0_fft = module.bvec_fft[0]
        bp_fft = module.bvec_fft[1]
        bm_fft = module.bvec_fft[2]
        energy_UU = (inner_prod(ux_fft, ux_fft) +
                     inner_prod(uy_fft, uy_fft) +
                     inner_prod(eta_fft, eta_fft) * self.sim.params.c2)
        energy_BB =  (inner_prod(b0_fft, b0_fft) +
                      inner_prod(bp_fft, bp_fft) +
                      inner_prod(bm_fft, bm_fft))
        self.assertTrue(np.allclose(energy_UU, energy_BB, rtol=0.1))
        
        Tq_tot_modes = 0.
        key_modes = ['Tq_GGG','Tq_AGG','Tq_GAAs','Tq_GAAd','Tq_AAA']
        for k in key_modes:
            Tq_tot_modes += dico_results[k]

        Tq_tot_exact = -self.sim.oper.spectrum2D_from_fft(
                         inner_prod(ux_fft, module.fnonlinfft_from_uxuy_funcfft(ux,uy,ux_fft)) +
                         inner_prod(uy_fft, module.fnonlinfft_from_uxuy_funcfft(ux,uy,uy_fft)))

        import ipdb; ipdb.set_trace()  
        self.assertTrue(np.allclose(Tq_tot_modes, Tq_tot_exact, rtol=0.1))

class TestSolvers(unittest.TestCase, TestOutput):
    def test_ns2d(self):
        """Should be able to run a NS2D simul."""
        self.sim = run_mini_simul('NS2D')
        clean_simul(self.sim)

    def test_sw1l(self):
        """Should be able to run a SW1L simul."""
        self.sim = run_mini_simul('SW1L', HAS_TO_SAVE=True)
        self.verify_sw1l_spect_energy_budg()
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
