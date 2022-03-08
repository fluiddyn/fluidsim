"""
SW1L forcing (:mod:`fluidsim.solvers.sw1l.forcing`)
===================================================

"""

import numpy as np

from fluiddyn.util import mpi

from fluidsim.base.forcing import ForcingBasePseudoSpectral

from fluidsim.base.forcing.specific import (
    InScriptForcingPseudoSpectral,
    Proportional as ProportionalBase,
    TimeCorrelatedRandomPseudoSpectral as TCRandomPS,
    RandomSimplePseudoSpectral,
)


class ForcingSW1L(ForcingBasePseudoSpectral):
    """Forcing class for the sw1l solver.

    .. inheritance-diagram:: ForcingSW1L

    """

    @staticmethod
    def _complete_info_solver(info_solver):
        """Complete the ParamContainer info_solver.

        This is a static method!
        """
        classes = [
            TCRandomPS,
            RandomSimplePseudoSpectral,
            Waves,
            WavesVortices,
            Potential,
        ]
        ForcingBasePseudoSpectral._complete_info_solver(info_solver, classes)


class TimeCorrelatedRandomPseudoSpectral(TCRandomPS):
    """Forces the geostrophic variable (by default) while maintaining a
    correlation for a certain time interval.

    """

    _key_forced_default = "q_fft"


class Proportional(ProportionalBase):
    _key_forced_default = "q_fft"


class Waves(RandomSimplePseudoSpectral):
    """Forces the ageostrophic variable and normalizes the forcing power based
    on the K.E and A.P.E thus generated. The forcing is white noise in time.

    """

    tag = "waves"
    _key_forced_default = "a_fft"

    def normalize_forcingc_2nd_degree_eq(self, Fa_fft, a_fft, key_forced=None):
        """Normalize the forcing Fa_fft such that the forcing rate of
        quadratic energy is equal to self.forcing_rate.
        """
        oper_c = self.oper_coarse
        params = self.params
        deltat = self.sim.time_stepping.deltat

        Fux_fft, Fuy_fft, Feta_fft = oper_c.uxuyetafft_from_afft(Fa_fft)
        ux_fft, uy_fft, eta_fft = oper_c.uxuyetafft_from_afft(a_fft)

        ax = deltat / 2 * oper_c.sum_wavenumbers(abs(Fux_fft) ** 2)
        ay = deltat / 2 * oper_c.sum_wavenumbers(abs(Fuy_fft) ** 2)
        aA = params.c2 * deltat / 2 * oper_c.sum_wavenumbers(abs(Feta_fft) ** 2)
        a = ax + ay + aA

        bx = oper_c.sum_wavenumbers((ux_fft.conj() * Fux_fft).real)
        by = oper_c.sum_wavenumbers((uy_fft.conj() * Fuy_fft).real)
        bA = params.c2 * oper_c.sum_wavenumbers((eta_fft.conj() * Feta_fft).real)
        b = bx + by + bA

        c = -self.forcing_rate
        Fa_fft *= self.coef_normalization_from_abc(a, b, c)


class WavesVortices(Waves):
    """Forces both the geostrophic and ageostrophic variable equally and
    normalizes the forcing rate based on the K.E and A.P.E thus generated.

    """

    tag = "waves_vortices"
    _key_forced_default = ("q_fft", "a_fft")

    def __init__(self, sim):
        super().__init__(sim)
        params = sim.params.forcing
        self.forcing_rate = 0.5 * params.forcing_rate

    def compute(self):
        if not isinstance(self.key_forced, (tuple, list, np.ndarray)):
            raise ValueError(
                "Expected array-like value for params.forcing.key_forced"
                + " = {} : {}".format(self.key_forced, type(self.key_forced))
            )

        v_fft = dict()
        Fv_fft = dict()

        for key in self.key_forced:
            try:
                v_fft[key] = self.sim.state.state_spect.get_var(key)
            except ValueError:
                v_fft[key] = self.sim.state.get_var(key)

            v_fft[key] = self.oper.coarse_seq_from_fft_loc(
                v_fft[key], self.shapeK_loc_coarse
            )

            if mpi.rank == 0:
                Fv_fft["F" + key] = self.forcingc_raw_each_time(key)

        if mpi.rank == 0:
            kwargs = v_fft
            kwargs.update(Fv_fft)
            Fv_fft = self.normalize_forcingc(**kwargs)
            self.fstate_coarse.init_statespect_from(**Fv_fft)

        self.put_forcingc_in_forcing()

    def normalize_forcingc(self, Fq_fft, Fa_fft, q_fft, a_fft):
        """Normalize the forcing Fa_fft such as the forcing rate of
        quadratic energy is equal to self.forcing_rate."""

        self.normalize_forcingc_2nd_degree_eq(Fa_fft, a_fft)

        oper_c = self.oper_coarse
        params = self.params
        deltat = self.sim.time_stepping.deltat

        Fux_fft, Fuy_fft, Feta_fft = oper_c.uxuyetafft_from_qfft(Fq_fft)
        ux_fft, uy_fft, eta_fft = oper_c.uxuyetafft_from_qfft(q_fft)

        ax = deltat / 2 * oper_c.sum_wavenumbers(abs(Fux_fft) ** 2)
        ay = deltat / 2 * oper_c.sum_wavenumbers(abs(Fuy_fft) ** 2)
        aA = params.c2 * deltat / 2 * oper_c.sum_wavenumbers(abs(Feta_fft) ** 2)
        a = ax + ay + aA

        bx = oper_c.sum_wavenumbers((ux_fft.conj() * Fux_fft).real)
        by = oper_c.sum_wavenumbers((uy_fft.conj() * Fuy_fft).real)
        bA = params.c2 * oper_c.sum_wavenumbers((eta_fft.conj() * Feta_fft).real)
        b = bx + by + bA

        c = -self.forcing_rate

        alpha = self.coef_normalization_from_abc(a, b, c)
        Fq_fft[:] = alpha * Fq_fft

        Fv_fft = {"q_fft": Fq_fft, "a_fft": Fa_fft}

        return Fv_fft


class Potential(Waves):
    """Forces only in A.P.E. and normalize for the desired forcing rate."""

    tag = "potential"
    _key_forced_default = "eta_fft"

    def normalize_forcingc_2nd_degree_eq(
        self, Feta_fft, eta_fft, key_forced=None
    ):
        """Normalize the forcing Fa_fft such as the forcing rate of
        quadratic energy is equal to self.forcing_rate."""
        if "eta_fft" not in self.key_forced:
            raise ValueError(
                "Expected 'eta_fft' in params.forcing.key_forced = {}".format(
                    self.key_forced
                )
            )

        oper_c = self.oper_coarse
        params = self.params
        deltat = self.sim.time_stepping.deltat

        a = params.c2 * deltat / 2 * oper_c.sum_wavenumbers(abs(Feta_fft) ** 2)

        b = params.c2 * oper_c.sum_wavenumbers((eta_fft.conj() * Feta_fft).real)

        c = -self.forcing_rate

        alpha = self.coef_normalization_from_abc(a, b, c)
        Feta_fft[:] = alpha * Feta_fft

        return Feta_fft


# class OldStuff(object):

#     def verify_injection_rate_opfft(self, q_fft, Fq_fft, oper):
#         """Verify injection rate."""
#         P_Z_forcing1 = abs(Fq_fft)**2/2*self.sim.time_stepping.deltat
#         P_Z_forcing2 = np.real(Fq_fft.conj()*q_fft)
#         P_Z_forcing1 = oper.sum_wavenumbers(P_Z_forcing1)
#         P_Z_forcing2 = oper.sum_wavenumbers(P_Z_forcing2)
#         if mpi.rank == 0:
#             print('P_Z_f = {0:9.4e} ; P_Z_f2 = {1:9.4e};'.format(
#                 P_Z_forcing1+P_Z_forcing2,
#                 P_Z_forcing2))

#     def verify_injection_rate_from_state(self):
#         """Verify injection rate."""

#         ux_fft = self.sim.state.state_spect.get_var('ux_fft')
#         uy_fft = self.sim.state.state_spect.get_var('uy_fft')
#         eta_fft = self.sim.state.state_spect.get_var('eta_fft')

#         q_fft, div_fft, ageo_fft = \
#             self.oper.qdafft_from_uxuyetafft(ux_fft, uy_fft, eta_fft)

#         Fux_fft = self.forcing_fft.get_var('ux_fft')
#         Fuy_fft = self.forcing_fft.get_var('uy_fft')
#         Feta_fft = self.forcing_fft.get_var('eta_fft')

#         Fq_fft, Fdiv_fft, Fageo_fft = \
#             self.oper.qdafft_from_uxuyetafft(Fux_fft, Fuy_fft, Feta_fft)
#         # print 'Fq_fft', abs(Fq_fft).max()
#         # print 'Fdiv_fft', abs(Fdiv_fft).max()
#         # print 'Fageo_fft', abs(Fageo_fft).max()

#         self.verify_injection_rate_opfft(q_fft, Fq_fft, self.oper)

#     def modify_Ffft_from_eta(self, F_fft, eta_fft):
#         """Put to zero the forcing for the too large modes."""
#         for ik in self.ind_forcing:
#             if abs(eta_fft.flat[ik]) > self.eta_cond:
#                 F_fft.flat[ik] = 0.
