from __future__ import print_function

import unittest
import numpy as np

from fluidsim.base.output.spect_energy_budget import inner_prod, cumsum_inv
from fluidsim.solvers.sw1l.output.test import BaseTestCase, mpi


debug = False


class TestSW1L(BaseTestCase):
    _tag = 'spect_energy_budg'
    exchange_keys = ['Cq_GG', 'Cq_AG', 'Cq_aG', 'Cq_AA']
    transfer_keys = ['Tq_GGG', 'Tq_AGG', 'Tq_GAAs', 'Tq_GAAd', 'Tq_AAA']

    @classmethod
    def setUpClass(cls, init_fields='dipole'):
        nh = 32
        super(TestSW1L, cls).setUpClass(
            nh=nh, init_fields=init_fields, HAS_TO_SAVE=True,
            forcing_enable=True)

    def skipUnlessHasAttr(self, attr, reason=None):
        attr_names = attr.split('.')

        attr = self
        while len(attr_names) > 0:
            subattr = attr_names.pop(0)
            if hasattr(attr, subattr):
                attr = getattr(attr, subattr)
            else:
                self.skipTest(reason)
                break

    def test_qmat(self):
        """Check qmat"""
        sim = self.sim
        module = self.output.spect_energy_budg
        self.skipUnlessHasAttr(
            'output.spect_energy_budg.norm_mode',
            self.solver + ' does not use normal mode spect_energy_budg')

        r, c, nkx, nky = module.norm_mode.qmat.shape
        identity = np.eye(r)
        for ikx in range(1, nkx):
            for iky in range(1, nky):
                qmat = module.norm_mode.qmat[:, :, ikx, iky]
                qct = qmat.conj().transpose()
                identity2 = np.dot(qct, qmat)
                try:
                    self.assertTrue(np.allclose(identity2, identity))
                except AssertionError:
                    print(('Q matrix identity not satisfied for kx, ky=',
                           sim.oper.KX[ikx, iky],
                           sim.oper.KY[ikx, iky]))
                    raise

    def test_energy_conservation(self):
        """ Check UU = BB energy conservation """
        sim = self.sim
        module = self.output.spect_energy_budg
        self.skipUnlessHasAttr(
            'output.spect_energy_budg.norm_mode',
            self.solver + " does not use normal mode spect_energy_budg")

        c2 = sim.params.c2
        get_var = sim.state.get_var
        ux_fft = get_var('ux_fft')
        uy_fft = get_var('uy_fft')
        eta_fft = get_var('eta_fft')
        b0_fft = module.norm_mode.bvec_fft[0]
        bp_fft = module.norm_mode.bvec_fft[1]
        bm_fft = module.norm_mode.bvec_fft[2]
        ux_fft[0, 0] = uy_fft[0, 0] = eta_fft[0, 0] = 0.
        energy_UU = (inner_prod(ux_fft, ux_fft) +
                     inner_prod(uy_fft, uy_fft) +
                     inner_prod(eta_fft, eta_fft) * c2)
        energy_BB = (inner_prod(b0_fft, b0_fft) +
                     inner_prod(bp_fft, bp_fft) +
                     inner_prod(bm_fft, bm_fft))
        np.testing.assert_allclose(
            energy_BB, energy_UU, rtol=1e-14, atol=1e-15)

    def test_decompositions(self):
        """Check normal mode decompositions."""
        sim = self.sim
        module = self.output.spect_energy_budg
        self.skipUnlessHasAttr(
            'output.spect_energy_budg.norm_mode',
            self.solver + " does not use normal mode spect_energy_budg")

        get_var = sim.state.get_var
        ux_fft = get_var('ux_fft')
        uy_fft = get_var('uy_fft')
        eta_fft = get_var('eta_fft')
        py_ux_fft = 1j * sim.oper.KY * ux_fft
        module.norm_mode.bvec_fft = module.norm_mode.bvecfft_from_uxuyetafft(
            ux_fft, uy_fft, eta_fft)

        key_modes, ux_fft_modes = module.norm_mode.normalmodefft_from_keyfft(
            'ux_fft')
        key_modes, uy_fft_modes = module.norm_mode.normalmodefft_from_keyfft(
            'uy_fft')
        key_modes, eta_fft_modes = module.norm_mode.normalmodefft_from_keyfft(
            'eta_fft')
        key_modes, py_ux_fft_modes = module.norm_mode.normalmodefft_from_keyfft(
            'py_ux_fft')
        ux_fft2 = uy_fft2 = eta_fft2 = py_ux_fft2 = 0.
        for mode in range(3):
            ux_fft2 += ux_fft_modes[mode]
            uy_fft2 += uy_fft_modes[mode]
            eta_fft2 += eta_fft_modes[mode]
            py_ux_fft2 += py_ux_fft_modes[mode]
        ux_fft[0, 0] = uy_fft[0, 0] = eta_fft[0, 0] = py_ux_fft[0, 0] = 0.
        atol = 1e-15
        rtol = 1e-14
        np.testing.assert_allclose(ux_fft2, ux_fft, rtol, atol)
        np.testing.assert_allclose(uy_fft2, uy_fft, rtol, atol)
        np.testing.assert_allclose(eta_fft2, eta_fft, rtol, atol)
        np.testing.assert_allclose(py_ux_fft2, py_ux_fft, rtol, atol)

    def test_exchange_term(self):
        """Check dyad decomposition. """

        sim = self.sim
        self.skipUnlessHasAttr(
            'output.spect_energy_budg.norm_mode',
            self.solver + " does not use normal mode spect_energy_budg")

        Cq_tot_modes = 0.
        for k in self.exchange_keys:
            Cq_tot_modes += self.dico[k]

        get_var = sim.state.get_var
        ux_fft = get_var('ux_fft')
        uy_fft = get_var('uy_fft')
        eta_fft = get_var('eta_fft')
        px_eta_fft, py_eta_fft = sim.oper.gradfft_from_fft(eta_fft)
        Cq_tot_exact = -sim.params.c2 * sim.oper.spectrum2D_from_fft(
            inner_prod(ux_fft, px_eta_fft) +
            inner_prod(uy_fft, py_eta_fft))
        np.testing.assert_allclose(
            Cq_tot_modes, Cq_tot_exact, rtol=1e-14, atol=1e-15)

    def test_transfer_term(self, check_hasattr=True,
                           include_non_quad_terms=True):
        """Check triad decomposition. """

        sim = self.sim
        module = self.output.spect_energy_budg
        if check_hasattr:
            self.skipUnlessHasAttr(
                'output.spect_energy_budg.norm_mode',
                self.solver + "does not use normal mode spect_energy_budg")

        Tq_tot_modes = 0.
        for k in self.transfer_keys:
            Tq_tot_modes += self.dico[k]

        get_var = sim.state.get_var
        ux_fft = get_var('ux_fft')
        uy_fft = get_var('uy_fft')
        eta_fft = get_var('eta_fft')
        ux = sim.state.state_phys.get_var('ux')
        uy = sim.state.state_phys.get_var('uy')
        eta = sim.state.state_phys.get_var('eta')
        TKq_exact = (
            inner_prod(ux_fft,
                       module.fnonlinfft_from_uxuy_funcfft(ux, uy, ux_fft)) +
            inner_prod(uy_fft,
                       module.fnonlinfft_from_uxuy_funcfft(ux, uy, uy_fft)))

        if include_non_quad_terms:
            div_fft = sim.oper.divfft_from_vecfft(ux_fft, uy_fft)
            div = sim.oper.ifft2(div_fft)
            divux_fft = sim.oper.fft2(div * ux)
            divuy_fft = sim.oper.fft2(div * uy)
            sim.oper.dealiasing(divux_fft, divuy_fft)
            TKdiv_exact = (inner_prod(ux_fft, divux_fft) +
                           inner_prod(uy_fft, divuy_fft)) * -0.5

            etaeta_fft = sim.oper.fft2(eta * eta)
            sim.oper.dealiasing(etaeta_fft)
            TPdiv_exact = 0.25 * sim.params.c2 * inner_prod(div_fft, etaeta_fft)
            TPq_exact_coef = -0.5 * sim.params.c2
        else:
            TKdiv_exact = 0.
            TPdiv_exact = 0.
            TPq_exact_coef = -sim.params.c2

        etaux_fft = sim.oper.fft2(eta * ux)
        etauy_fft = sim.oper.fft2(eta * uy)
        sim.oper.dealiasing(etaux_fft, etauy_fft)
        TPq_exact = TPq_exact_coef * inner_prod(
            eta_fft,
            sim.oper.divfft_from_vecfft(etaux_fft, etauy_fft))
        Tq_tot_exact = sim.oper.spectrum2D_from_fft(
            TKq_exact + TKdiv_exact + TPdiv_exact + TPq_exact)

        if debug:
            Tq_abs_error = np.absolute(Tq_tot_modes - Tq_tot_exact)
            Tq_rel_error = Tq_abs_error[Tq_tot_exact > 0] / Tq_tot_exact[Tq_tot_exact > 0]
            msg = 'abs. error max: {}, min: {}; '.format(
                Tq_abs_error.max(), Tq_abs_error[Tq_abs_error > 0].min())
            msg += 'rel. error max: {}, min: {}'.format(
                Tq_rel_error.max(), Tq_rel_error.min())
            print(sim.__class__, msg)
        else:
            msg = ''

        np.testing.assert_allclose(
            Tq_tot_modes, Tq_tot_exact, rtol=1e-8, atol=1e-9, err_msg=msg)

    def test_triad_conservation_laws(self):
        r"""Tests for certain energy and enstrophy conservation laws.

        .. math:: \Sigma T_{GGG} = 0
        .. math:: k^{2}\Sigma T_{GGG} = 0
        """
        sim = self.sim
        try:
            Tq_GGG = self.dico['Tq_GGG']
            Tens = self.dico['Tens']
        except KeyError:
            Tq_GGG = self.dico['transfer2D_Errr']
            Tens = self.dico['transfer2D_CPE']

        energy_GGG = Tq_GGG.sum()
        enstrophy_GGG = Tens.sum()

        self.assertAlmostEqual(energy_GGG, 0)
        self.assertAlmostEqual(enstrophy_GGG, 0)

        if mpi.nb_proc == 1:
            dkh = sim.oper.deltakh
            Pi_GGG = cumsum_inv(Tq_GGG) * dkh
            Pi_ens = cumsum_inv(Tens) * dkh
            energy_GGG = Pi_GGG[0]
            enstrophy_GGG = Pi_ens[0]

            self.assertAlmostEqual(energy_GGG, 0)
            self.assertAlmostEqual(enstrophy_GGG, 0)


class TestWaves(TestSW1L):
    solver = 'sw1l.onlywaves'
    # exchange_keys = ['convP2D', 'convK2D']
    # transfer_keys = [
    #     'transfer2D_Errr', 'transfer2D_Edrd', 'transfer2D_Edrr_rrd',
    #     'transfer2D_Eureu',  # K.E. transfer terms
    #     'transfer2D_Erdr', 'transfer2D_Eddd', 'transfer2D_Eddr_rdd',
    #     'transfer2D_Eudeu'  # P.E. transfer terms
    # ]

    @classmethod
    def setUpClass(cls):
        super(TestWaves, cls).setUpClass(init_fields='noise')

    def test_transfer_term(self):
        """Check triad decomposition. """
        super(TestWaves, self).test_transfer_term()
        #     check_hasattr=False, include_non_quad_terms=False)


class TestExactlin(TestSW1L):
    solver = 'sw1l.exactlin'

    @unittest.skipIf(mpi.nb_proc > 1,
                     'plot function works sequentially only')
    def test_plot_spect_energy_budg(self):
        self._plot()

    def test_online_plot_spect_energy_budg(self):
        self._online_plot_saving(self.dico)


class TestExmod(TestSW1L):
    solver = 'sw1l.exactlin.modified'

    def test_transfer_term(self, check_hasattr=True):
        """Check triad decomposition. """

        sim = self.sim
        module = self.output.spect_energy_budg
        if check_hasattr:
            self.skipUnlessHasAttr(
                'output.spect_energy_budg.norm_mode',
                self.solver + "does not use normal mode spect_energy_budg")

        Tq_tot_modes = 0.
        for k in self.transfer_keys:
            Tq_tot_modes += self.dico[k]

        get_var = sim.state.get_var
        ux_fft = get_var('ux_fft')
        uy_fft = get_var('uy_fft')
        eta_fft = get_var('eta_fft')
        rot_fft = sim.oper.rotfft_from_vecfft(ux_fft, uy_fft)

        urx_fft, ury_fft = sim.oper.vecfft_from_rotfft(rot_fft)
        urx = sim.oper.ifft2(urx_fft)
        ury = sim.oper.ifft2(ury_fft)
        TKq_exact = (
            inner_prod(ux_fft,
                       module.fnonlinfft_from_uxuy_funcfft(urx, ury, ux_fft)) +
            inner_prod(uy_fft,
                       module.fnonlinfft_from_uxuy_funcfft(urx, ury, uy_fft)))
        TPq_exact = sim.params.c2 * (
            inner_prod(eta_fft,
                       module.fnonlinfft_from_uxuy_funcfft(urx, ury, eta_fft)))

        Tq_tot_exact = sim.oper.spectrum2D_from_fft(
            TKq_exact + TPq_exact)

        if debug:
            Tq_abs_error = np.absolute(Tq_tot_modes - Tq_tot_exact)
            Tq_rel_error = Tq_abs_error[Tq_tot_exact > 0] / Tq_tot_exact[Tq_tot_exact > 0]
            msg = 'abs. error max: {}, min: {}; '.format(
                Tq_abs_error.max(), Tq_abs_error[Tq_abs_error > 0].min())
            msg += 'rel. error max: {}, min: {}'.format(
                Tq_rel_error.max(), Tq_rel_error.min())
            print(sim.__class__, msg)
        else:
            msg = ''

        np.testing.assert_allclose(
            Tq_tot_modes, Tq_tot_exact, rtol=1e-8, atol=1e-9, err_msg=msg)


class TestModif(TestExmod):
    solver = 'sw1l.modified'
    exchange_keys = ['convA2D']
    transfer_keys = [
        'transfer2D_Errr', 'transfer2D_Edrd', 'transfer2D_Edrr_rrd',  # K.E.
        'transfer2D_EA',  # P.E.
    ]

    def test_transfer_term(self):
        """Check triad decomposition. """
        super(TestModif, self).test_transfer_term(check_hasattr=False)

    @unittest.skip('Uses the same plot function as sw1l')
    def test_plot_spect_energy_budg(self):
        pass

    @unittest.skip('Uses the same plot function as sw1l')
    def test_online_plot_spect_energy_budg(self):
        pass


if __name__ == '__main__':
    unittest.main()
