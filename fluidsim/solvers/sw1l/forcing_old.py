"""
SW1L forcing (:mod:`fluidsim.solvers.sw1l.forcing`)
===================================================

"""

import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.forcing import ForcingBasePseudoSpectral

from fluidsim.base.forcing.specific import \
    Proportional as ProportionalBase

from fluidsim.base.forcing.specific import \
    TimeCorrelatedRandomPseudoSpectral as TCRandomPS


class ForcingSW1L(ForcingBasePseudoSpectral):

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        classes = [Proportional, TimeCorrelatedRandomPseudoSpectral]
        ForcingBasePseudoSpectral._complete_info_solver(info_solver, classes)


class TCRandomPSW(TCRandomPS):
    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the *params* container."""
        super(TCRandomPSW, cls)._complete_params_with_default(params)
        params.forcing.key_forced = 'q_fft'


class Proportional(ProportionalBase):
    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the *params* container."""
        super(Proportional, cls)._complete_params_with_default(params)
        params.forcing.key_forced = 'q_fft'


class TimeCorrelatedRandomPseudoSpectral(TCRandomPSW):
    def compute_forcingc_raw(self):
        Fw_fft = super(TimeCorrelatedRandomPseudoSpectral,
                       self).compute_forcingc_raw()

        return Fw_fft

    # def compute_forcing_proportional(self):
    #     """Compute a forcing proportional to the flow."""
    #     shapeK_loc_c = self.shapeK_loc_coarse
    #     q_fft = self.qfftcoarse_from_setvarfft()

    #     if mpi.rank > 0:
    #         Fq_fft = np.empty(shapeK_loc_c,
    #                           dtype=np.complex128)
    #     else:
    #         Fq_fft = self.normalize_forcingc_proportional(q_fft)
    #         # self.verify_injection_rate_opfft(q_fft, Fq_fft,
    #         #                                  self.oper_coarse)
    #         self.fill_forcingc_from_Fqfft(Fq_fft)

    #     self.put_forcingc_in_forcing()
    #     ## verification
    #     self.verify_injection_rate_from_state()

    # def compute_forcing_2nd_degree_eq(self):
    #     """compute a forcing normalized with a 2nd degree eq."""
    #     shapeK_loc_c = self.shapeK_loc_coarse
    #     q_fft = self.qfftcoarse_from_setvarfft()

    #     if mpi.rank > 0:
    #         Fq_fft = np.empty(shapeK_loc_c,
    #                           dtype=np.complex128)
    #     else:
    #         Fq_fft = self.forcingc_raw_each_time()
    #         Fq_fft = self.normalize_forcingc_2nd_degree_eq(Fq_fft,
    #                                                        q_fft)
    #         # self.verify_injection_rate_opfft(q_fft, Fq_fft,
    #         #                                  self.oper_coarse)
    #         self.fill_forcingc_from_Fqfft(Fq_fft)

    #     self.put_forcingc_in_forcing()
    #     ## verification
    #     # self.verify_injection_rate_from_state()


class OldStuff(object):

    def compute_forcing_waves(self):
        """compute a forcing normalized with a 2nd degree eq."""
        shapeK_loc_c = self.shapeK_loc_coarse
        a_fft, eta_fft = self.aetafftcoarse_from_setvarfft()
        if mpi.rank > 0:
            Fa_fft = np.empty(shapeK_loc_c,
                              dtype=np.complex128)
        else:
            Fa_fft = self.forcingc_random()
            self.modify_Ffft_from_eta(Fa_fft, eta_fft)

            if np.max(abs(Fa_fft)) > 0:
                self.normalize_Fafft_constPquadE(Fa_fft,
                                                 a_fft)

            self.fill_forcingc_from_Fafft(Fa_fft)

        self.put_forcingc_in_forcing()

    def compute_forcing_particular_k(self):
        """compute a forcing "decorralated" from the flow"""

        shapeK_loc_c = self.shapeK_loc_coarse
        q_fft = self.qfftcoarse_from_setvarfft()

        if mpi.rank > 0:
            Fq_fft = np.empty(shapeK_loc_c,
                              dtype=np.complex128)
        else:
            Fq_fft = self.forcingc_raw_each_time()
            Fq_fft = self.normalize_forcingc_part_k(Fq_fft,
                                                    q_fft)
            # self.verify_injection_rate_opfft(q_fft, Fq_fft,
            #                                  self.oper_coarse)
            self.fill_forcingc_from_Fqfft(Fq_fft)

        self.put_forcingc_in_forcing()
        # verification
        self.verify_injection_rate_from_state()

    def verify_injection_rate_opfft(self, q_fft, Fq_fft, oper):
        """Verify injection rate."""
        P_Z_forcing1 = abs(Fq_fft)**2/2*self.sim.time_stepping.deltat
        P_Z_forcing2 = np.real(Fq_fft.conj()*q_fft)
        P_Z_forcing1 = oper.sum_wavenumbers(P_Z_forcing1)
        P_Z_forcing2 = oper.sum_wavenumbers(P_Z_forcing2)
        if mpi.rank == 0:
            print 'P_Z_f = {0:9.4e} ; P_Z_f2 = {1:9.4e};'.format(
                P_Z_forcing1+P_Z_forcing2,
                P_Z_forcing2)

    def verify_injection_rate_from_state(self):
        """Verify injection rate."""

        ux_fft = self.sim.state.state_fft.get_var('ux_fft')
        uy_fft = self.sim.state.state_fft.get_var('uy_fft')
        eta_fft = self.sim.state.state_fft.get_var('eta_fft')

        q_fft, div_fft, ageo_fft = \
            self.oper.qdafft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

        Fux_fft = self.forcing_fft.get_var('ux_fft')
        Fuy_fft = self.forcing_fft.get_var('uy_fft')
        Feta_fft = self.forcing_fft.get_var('eta_fft')

        Fq_fft, Fdiv_fft, Fageo_fft = \
            self.oper.qdafft_from_uxuyetafft(Fux_fft, Fuy_fft, Feta_fft)
        # print 'Fq_fft', abs(Fq_fft).max()
        # print 'Fdiv_fft', abs(Fdiv_fft).max()
        # print 'Fageo_fft', abs(Fageo_fft).max()

        self.verify_injection_rate_opfft(q_fft, Fq_fft, self.oper)

    def qfftcoarse_from_setvarfft(self, set_var_fft=None):
        if set_var_fft is None:
            set_var_fft = self.sim.state.state_fft
        ux_fft = set_var_fft.get_var('ux_fft')
        uy_fft = set_var_fft.get_var('uy_fft')
        eta_fft = set_var_fft.get_var('eta_fft')
        shapeK_loc_c = self.shapeK_loc_coarse
        ux_fft = self.oper.coarse_seq_from_fft_loc(ux_fft, shapeK_loc_c)
        uy_fft = self.oper.coarse_seq_from_fft_loc(uy_fft, shapeK_loc_c)
        eta_fft = self.oper.coarse_seq_from_fft_loc(eta_fft, shapeK_loc_c)
        if mpi.rank > 0:
            q_fft = np.empty(shapeK_loc_c,
                             dtype=np.complex128)
        else:
            rot_fft = self.oper_coarse.rotfft_from_vecfft(ux_fft, uy_fft)
            q_fft = rot_fft-self.params.f*eta_fft
        return q_fft

    def etafftcoarse_from_setvarfft(self, set_var_fft=None):
        if set_var_fft is None:
            set_var_fft = self.sim.state.state_fft
        eta_fft = set_var_fft.get_var('eta_fft')
        shapeK_loc_c = self.shapeK_loc_coarse
        eta_fft = self.oper.coarse_seq_from_fft_loc(eta_fft, shapeK_loc_c)
        return eta_fft

    def aetafftcoarse_from_setvarfft(self, set_var_fft=None):
        if set_var_fft is None:
            set_var_fft = self.sim.state.state_fft
        eta_fft = set_var_fft.get_var('eta_fft')
        ux_fft = set_var_fft.get_var('ux_fft')
        uy_fft = set_var_fft.get_var('uy_fft')
        shapeK_loc_c = self.shapeK_loc_coarse
        eta_fft = self.oper.coarse_seq_from_fft_loc(eta_fft, shapeK_loc_c)
        ux_fft = self.oper.coarse_seq_from_fft_loc(ux_fft, shapeK_loc_c)
        uy_fft = self.oper.coarse_seq_from_fft_loc(uy_fft, shapeK_loc_c)

        if mpi.rank > 0:
            a_fft = np.empty(shapeK_loc_c,
                             dtype=np.complex128)
        else:
            a_fft = self.oper_coarse.afft_from_uxuyetafft(
                ux_fft, uy_fft, eta_fft)

        return a_fft, eta_fft

    def modify_Ffft_from_eta(self, F_fft, eta_fft):
        """Put to zero the forcing for the too large modes."""
        for ik in self.ind_forcing:
            if abs(eta_fft.flat[ik]) > self.eta_cond:
                F_fft.flat[ik] = 0.

    def fill_forcingc_from_Fqfft(self, Fq_fft):

        Fux_fft, Fuy_fft, Feta_fft = \
            self.oper_coarse.uxuyetafft_from_qfft(Fq_fft)
        self.forcingc_fft.set_var('ux_fft', Fux_fft)
        self.forcingc_fft.set_var('uy_fft', Fuy_fft)
        self.forcingc_fft.set_var('eta_fft', Feta_fft)

    def fill_forcingc_from_Fetafft(self, Feta_fft):

        self.forcingc_fft.set_var(
            'ux_fft', self.oper_coarse.constant_arrayK(value=0.))
        self.forcingc_fft.set_var(
            'uy_fft', self.oper_coarse.constant_arrayK(value=0.))
        self.forcingc_fft.set_var('eta_fft', Feta_fft)

    def fill_forcingc_from_Fafft(self, Fa_fft):

        Fux_fft, Fuy_fft, Feta_fft = \
            self.oper_coarse.uxuyetafft_from_afft(Fa_fft)
        self.forcingc_fft.set_var('ux_fft', Fux_fft)
        self.forcingc_fft.set_var('uy_fft', Fuy_fft)
        self.forcingc_fft.set_var('eta_fft', Feta_fft)

    def get_FxFyFetafft(self):

        Fx_fft = self.forcing_fft.get_var('ux_fft')
        Fy_fft = self.forcing_fft.get_var('uy_fft')
        Feta_fft = self.forcing_fft.get_var('eta_fft')
        return Fx_fft, Fy_fft, Feta_fft

    def normalize_Fafft_constPquadE(self, Fa_fft, a_fft):
        """Normalize the forcing Fa_fft such as the forcing rate of
        quadratic energy is equal to self.forcing_rate."""
        oper_c = self.oper_coarse
        params = self.params
        deltat = self.sim.time_stepping.deltat

        Fux_fft, Fuy_fft, Feta_fft = \
            oper_c.uxuyetafft_from_afft(Fa_fft)
        ux_fft, uy_fft, eta_fft = \
            oper_c.uxuyetafft_from_afft(a_fft)

        ax = deltat/2*oper_c.sum_wavenumbers(abs(Fux_fft)**2)
        ay = deltat/2*oper_c.sum_wavenumbers(abs(Fuy_fft)**2)
        aA = params.c2*deltat/2*oper_c.sum_wavenumbers(abs(Feta_fft)**2)
        a = ax + ay + aA

        bx = oper_c.sum_wavenumbers(
            (ux_fft.conj()*Fux_fft).real)
        by = oper_c.sum_wavenumbers(
            (uy_fft.conj()*Fuy_fft).real)
        bA = params.c2*oper_c.sum_wavenumbers(
            (eta_fft.conj()*Feta_fft).real)
        b = bx + by + bA

        c = -self.forcing_rate

        Delta = b**2 - 4*a*c
        alpha = (np.sqrt(Delta) - b)/(2*a)

        Fa_fft[:] = alpha*Fa_fft


class ForcingSW1LExactLin(ForcingSW1L):

    def verify_injection_rate_from_state(self):
        """Verify injection rate."""

        q_fft = self.sim.state.state_fft.get_var('q_fft')
        Fq_fft = self.forcing_fft.get_var('q_fft')
        # print 'Fq_fft', abs(Fq_fft).max()
        self.verify_injection_rate_opfft(q_fft, Fq_fft, self.oper)

    def qfftcoarse_from_setvarfft(self, set_var_fft=None):
        if set_var_fft is None:
            set_var_fft = self.sim.state.state_fft
        q_fft = set_var_fft.get_var('q_fft')
        shapeK_loc_c = self.shapeK_loc_coarse
        q_fft = self.oper.coarse_seq_from_fft_loc(q_fft, shapeK_loc_c)
        return q_fft

    def fill_forcingc_from_Fqfft(self, Fq_fft):

        self.forcingc_fft.set_var('q_fft', Fq_fft)
        self.forcingc_fft.set_var(
            'ap_fft', self.oper_coarse.constant_arrayK(value=0.))
        self.forcingc_fft.set_var(
            'am_fft', self.oper_coarse.constant_arrayK(value=0.))

    def get_FxFyFetafft(self):

        Fq_fft = self.forcing_fft.get_var('q_fft')
        Fp_fft = self.forcing_fft.get_var('ap_fft')
        Fm_fft = self.forcing_fft.get_var('am_fft')

        return self.oper.uxuyetafft_from_qapamfft(Fq_fft, Fp_fft, Fm_fft)

    def etafftcoarse_from_setvarfft(self, set_var_fft=None):
        if set_var_fft is None:
            set_var_fft = self.sim.state.state_fft
        q_fft = set_var_fft.get_var('q_fft')
        ap_fft = set_var_fft.get_var('ap_fft')
        am_fft = set_var_fft.get_var('am_fft')
        a_fft = ap_fft + am_fft
        eta_fft = (self.oper_coarse.etafft_from_qfft(q_fft)
                   + self.oper_coarse.etafft_from_afft(a_fft)
                   )
        shapeK_loc_c = self.shapeK_loc_coarse
        eta_fft = self.oper.coarse_seq_from_fft_loc(eta_fft, shapeK_loc_c)
        return eta_fft

    def fill_forcingc_from_Fafft(self, Fa_fft):

        self.forcingc_fft.set_var(
            'q_fft', self.oper_coarse.constant_arrayK(value=0.))
        self.forcingc_fft.set_var('ap_fft', 0.5*Fa_fft)
        self.forcingc_fft.set_var('am_fft', 0.5*Fa_fft)

    def aetafftcoarse_from_setvarfft(self, set_var_fft=None):
        if set_var_fft is None:
            set_var_fft = self.sim.state.state_fft
        q_fft = set_var_fft.get_var('q_fft')
        ap_fft = set_var_fft.get_var('ap_fft')
        am_fft = set_var_fft.get_var('am_fft')
        shapeK_loc_c = self.shapeK_loc_coarse
        q_fft = self.oper.coarse_seq_from_fft_loc(q_fft, shapeK_loc_c)
        ap_fft = self.oper.coarse_seq_from_fft_loc(ap_fft, shapeK_loc_c)
        am_fft = self.oper.coarse_seq_from_fft_loc(am_fft, shapeK_loc_c)

        if mpi.rank > 0:
            a_fft = np.empty(shapeK_loc_c,
                             dtype=np.complex128)
            eta_fft = np.empty(shapeK_loc_c,
                               dtype=np.complex128)
        else:
            a_fft = ap_fft + am_fft
            eta_fft = self.oper_coarse.etafft_from_aqfft(a_fft, q_fft)

        return a_fft, eta_fft


class ForcingSW1LWaves(ForcingSW1L):

    def qfftcoarse_from_setvarfft(self, set_var_fft=None):
        raise ValueError('This solver does not solve for q.')


    def get_FxFyFetafft(self):

        Fq_fft = self.oper.constant_arrayK(value=0)
        Fp_fft = self.forcing_fft.get_var('ap_fft')
        Fm_fft = self.forcing_fft.get_var('am_fft')

        return self.oper.uxuyetafft_from_qapamfft(Fq_fft, Fp_fft, Fm_fft)

    def etafftcoarse_from_setvarfft(self, set_var_fft=None):
        if set_var_fft is None:
            set_var_fft = self.sim.state.state_fft
        ap_fft = set_var_fft.get_var('ap_fft')
        am_fft = set_var_fft.get_var('am_fft')
        a_fft = ap_fft + am_fft
        eta_fft = self.oper_coarse.etafft_from_afft(a_fft)
        shapeK_loc_c = self.shapeK_loc_coarse
        eta_fft = self.oper.coarse_seq_from_fft_loc(eta_fft, shapeK_loc_c)
        return eta_fft

    def fill_forcingc_from_Fafft(self, Fa_fft):

        self.forcingc_fft.set_var('ap_fft', 0.5*Fa_fft)
        self.forcingc_fft.set_var('am_fft', 0.5*Fa_fft)

    def aetafftcoarse_from_setvarfft(self, set_var_fft=None):
        if set_var_fft is None:
            set_var_fft = self.sim.state.state_fft
        ap_fft = set_var_fft.get_var('ap_fft')
        am_fft = set_var_fft.get_var('am_fft')
        shapeK_loc_c = self.shapeK_loc_coarse
        ap_fft = self.oper.coarse_seq_from_fft_loc(ap_fft, shapeK_loc_c)
        am_fft = self.oper.coarse_seq_from_fft_loc(am_fft, shapeK_loc_c)

        if mpi.rank > 0:
            a_fft = np.empty(shapeK_loc_c,
                             dtype=np.complex128)
            eta_fft = np.empty(shapeK_loc_c,
                               dtype=np.complex128)
        else:
            a_fft = ap_fft + am_fft
            q_fft = self.oper_coarse.constant_arrayK(value=0)
            eta_fft = self.oper_coarse.etafft_from_aqfft(a_fft, q_fft)

        return a_fft, eta_fft
