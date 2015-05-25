"""
SW1L forcing (:mod:`fluidsim.solvers.sw1l.forcing`)
===================================================

"""

import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.forcing import ForcingBasePseudoSpectral

from fluidsim.base.forcing.specific import \
    Proportional as ProportionalBase

from fluidsim.base.forcing.specific import (
    TimeCorrelatedRandomPseudoSpectral as TCRandomPS,
    RamdomSimplePseudoSpectral)


class ForcingSW1L(ForcingBasePseudoSpectral):
    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        classes = [Proportional, TimeCorrelatedRandomPseudoSpectral, Waves]
        ForcingBasePseudoSpectral._complete_info_solver(info_solver, classes)


class TimeCorrelatedRandomPseudoSpectral(TCRandomPS):
    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the *params* container."""
        super(TimeCorrelatedRandomPseudoSpectral,
              cls)._complete_params_with_default(params)
        params.forcing.key_forced = 'q_fft'


class Proportional(ProportionalBase):
    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the *params* container."""
        super(Proportional, cls)._complete_params_with_default(params)
        params.forcing.key_forced = 'q_fft'


class Waves(RamdomSimplePseudoSpectral):
    tag = 'waves'

    @classmethod
    def _complete_params_with_default(cls, params):
        """Complete the *params* container."""
        super(Waves, cls)._complete_params_with_default(params)
        params.forcing.key_forced = 'a_fft'

    def normalize_forcingc_2nd_degree_eq(self, Fa_fft, a_fft):
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

        return Fa_fft


class OldStuff(object):

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

    def modify_Ffft_from_eta(self, F_fft, eta_fft):
        """Put to zero the forcing for the too large modes."""
        for ik in self.ind_forcing:
            if abs(eta_fft.flat[ik]) > self.eta_cond:
                F_fft.flat[ik] = 0.
