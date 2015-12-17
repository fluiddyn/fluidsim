import unittest
import numpy as np
from fluidsim.base.output.spect_energy_budget import inner_prod, cumsum_inv
from fluidsim.solvers.test.test_solvers import run_mini_simul, clean_simul

sim = run_mini_simul('SW1L', HAS_TO_SAVE=True)
module = sim.output.spect_energy_budg
dico_results = module.compute()


class TestSpectEnergyBudg(unittest.TestCase):

    def test_qmat(self):
        """Check qmat"""
        
        r, c, nkx, nky = module.qmat.shape
        identity = np.eye(r)
        for ikx in xrange(1,nkx):
            for iky in xrange(1,nky):
                qmat = module.qmat[:,:,ikx,iky]
                qct = qmat.conj().transpose()
                identity2 = np.dot(qct,qmat)
                try:
                    self.assertTrue(np.allclose(identity2, identity))
                except AssertionError:
                    print('Q matrix identity not satisfied for kx, ky=',
                            sim.oper.KX[ikx,iky],
                            sim.oper.KY[ikx,iky])
                    raise
    
    def test_energy_conservation(self):
        """ Check UU = BB energy conservation """
        
        c2 = sim.params.c2
        ux_fft = sim.state('ux_fft')
        uy_fft = sim.state('uy_fft')
        eta_fft = sim.state('eta_fft')
        b0_fft = module.bvec_fft[0]
        bp_fft = module.bvec_fft[1]
        bm_fft = module.bvec_fft[2]
        ux_fft[0,0] = uy_fft[0,0] = eta_fft[0,0] = 0.
        energy_UU = (inner_prod(ux_fft, ux_fft) +
                     inner_prod(uy_fft, uy_fft) +
                     inner_prod(eta_fft, eta_fft) * c2)
        energy_BB =  (inner_prod(b0_fft, b0_fft) +
                      inner_prod(bp_fft, bp_fft) +
                      inner_prod(bm_fft, bm_fft))
        self.assertTrue(np.allclose(energy_BB, energy_UU))
    
    def test_decompositions(self):
        """ Check normal mode, dyad and triad decompositions """

        ux_fft = sim.state('ux_fft')
        uy_fft = sim.state('uy_fft')
        eta_fft = sim.state('eta_fft')
        ux = sim.state.state_phys.get_var('ux')
        uy = sim.state.state_phys.get_var('uy')
        eta = sim.state.state_phys.get_var('eta')
        py_ux_fft = 1j * sim.oper.KY * ux_fft
        module.bvec_fft = module.bvecfft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        key_modes, ux_fft_modes = module._normalmodefft_from_keyfft('ux_fft')
        key_modes, uy_fft_modes = module._normalmodefft_from_keyfft('uy_fft')
        key_modes, eta_fft_modes = module._normalmodefft_from_keyfft('eta_fft')
        key_modes, py_ux_fft_modes = module._normalmodefft_from_keyfft('py_ux_fft')
        ux_fft2 = uy_fft2 = eta_fft2 = py_ux_fft2 = 0. 
        for mode in xrange(3):
            ux_fft2 += ux_fft_modes[mode]
            uy_fft2 += uy_fft_modes[mode]
            eta_fft2 += eta_fft_modes[mode]
            py_ux_fft2 += py_ux_fft_modes[mode]
        ux_fft[0,0] = uy_fft[0,0] = eta_fft[0,0] = py_ux_fft[0,0] = 0.
        self.assertTrue(np.allclose(ux_fft2, ux_fft))
        self.assertTrue(np.allclose(uy_fft2, uy_fft))
        self.assertTrue(np.allclose(eta_fft2, eta_fft))
        self.assertTrue(np.allclose(py_ux_fft2, py_ux_fft))

        # Check dyad decomposition
        Cq_tot_modes = 0.
        key_modes = ['Cq_GG','Cq_AG','Cq_aG','Cq_AA']
        for k in key_modes:
            Cq_tot_modes += dico_results[k]

        px_eta_fft, py_eta_fft = sim.oper.gradfft_from_fft(eta_fft)
        Cq_tot_exact =  -sim.params.c2 * sim.oper.spectrum2D_from_fft(
                         inner_prod(ux_fft, px_eta_fft) +
                         inner_prod(uy_fft, py_eta_fft))
        self.assertTrue(np.allclose(Cq_tot_exact, Cq_tot_modes))
        
        # Check triad decomposition
        Tq_tot_modes = 0.
        key_modes = ['Tq_GGG','Tq_AGG','Tq_GAAs','Tq_GAAd','Tq_AAA']
        for k in key_modes:
            Tq_tot_modes += dico_results[k]
        
        TKq_exact = (inner_prod(ux_fft, module.fnonlinfft_from_uxuy_funcfft(ux,uy,ux_fft)) +
                     inner_prod(uy_fft, module.fnonlinfft_from_uxuy_funcfft(ux,uy,uy_fft)))
        
        div = sim.oper.ifft2(sim.oper.divfft_from_vecfft(ux_fft, uy_fft))
        divux_fft = sim.oper.fft2(div * ux)
        divuy_fft = sim.oper.fft2(div * uy)
        sim.oper.dealiasing(divux_fft, divuy_fft)
        TKdiv_exact = (inner_prod(ux_fft, divux_fft) +
                       inner_prod(uy_fft, divuy_fft)) * -0.5

        etaux_fft = sim.oper.fft2(eta * ux)
        etauy_fft = sim.oper.fft2(eta * uy)
        sim.oper.dealiasing(etaux_fft, etauy_fft)
        TPq_exact = - sim.params.c2 * inner_prod(eta_fft,
                                               sim.oper.divfft_from_vecfft(etaux_fft, etauy_fft))
        Tq_tot_exact = sim.oper.spectrum2D_from_fft(TKq_exact + TKdiv_exact + TPq_exact)
        
        self.assertTrue(np.allclose(Tq_tot_exact, Tq_tot_modes))

    def test_triad_conservation_laws(self):
        r"""
        Tests for certain energy and enstrophy conservation laws.

        .. math:: \Sigma T_{GGG} = 0
        .. math:: k^{2}\Sigma T_{GGG} = 0
        """

        Tq_GGG = dico_results['Tq_GGG']
        Tens = dico_results['Tens']

        energy_GGG = Tq_GGG.sum()
        enstrophy_GGG = Tens.sum()

        self.assertAlmostEqual(energy_GGG,0)
        self.assertAlmostEqual(enstrophy_GGG,0)
        
        dkh = sim.oper.deltakh
        Pi_GGG = cumsum_inv(Tq_GGG) * dkh
        Pi_ens = cumsum_inv(Tens) * dkh
        energy_GGG = Pi_GGG[0]
        enstrophy_GGG = Pi_ens[0]
        
        self.assertAlmostEqual(energy_GGG,0)
        self.assertAlmostEqual(enstrophy_GGG,0)


if __name__ == '__main__':
    unittest.main()
    clean_simul(sim)
