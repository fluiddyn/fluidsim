"""Forcing schemes (:mod:`fluidsim.base.forcing.specific`)
================================================================


Provides:

.. autoclass:: SpecificForcing
   :members:
   :private-members:

.. autoclass:: SpecificForcingPseudoSpectral
   :members:
   :private-members:

.. autoclass:: InScriptForcingPseudoSpectral
   :members:
   :private-members:

.. autoclass:: NormalizedForcing
   :members:
   :private-members:

.. autoclass:: Proportional
   :members:
   :private-members:

.. autoclass:: RandomSimplePseudoSpectral
   :members:
   :private-members:

.. autoclass:: TimeCorrelatedRandomPseudoSpectral
   :members:
   :private-members:

.. autoclass:: TimeCorrelatedRandomPseudoSpectralAnisotropic
   :members:
   :private-members:

"""
from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object

from copy import deepcopy
from math import radians
import types

import numpy as np

from fluiddyn.util import mpi
from fluidsim.base.setofvariables import SetOfVariables


class SpecificForcing(object):
    """Base class for specific forcing"""
    tag = 'specific'

    @classmethod
    def _complete_params_with_default(cls, params):
        params.forcing.available_types.append(cls.tag)

    def __init__(self, sim):

        self.sim = sim
        self.oper = sim.oper
        self.params = sim.params


class SpecificForcingPseudoSpectral(SpecificForcing):
    """Specific forcing for pseudo-spectra solvers"""

    tag = 'pseudo_spectral'
    _key_forced_default = 'rot_fft'

    @staticmethod
    def _check_forcing_shape(shape_forcing, shape):
        """Check if shape of the forcing array exceeds the shape
        of the global array.

        Parameters
        ----------
        shape_forcing: array-like
            A single-element array containing index of largest forcing
            wavenumber or a tuple indincating shape of the forcing array.

        shape: array-like
            A tuple indicating the shape of an array or Operators instance.

        Returns
        -------
        bool

        """
        if any(np.greater(shape_forcing, shape)):
            raise NotImplementedError(
                'The resolution is too small for the required forcing: '
                'any{} < {}'.format(shape, shape_forcing))

    def __init__(self, sim):

        super(SpecificForcingPseudoSpectral, self).__init__(sim)

        params = sim.params

        self.sum_wavenumbers = sim.oper.sum_wavenumbers
        self.fft2 = sim.oper.fft2
        self.ifft2 = sim.oper.ifft2

        self.forcing_fft = SetOfVariables(
            like=sim.state.state_spect, info='forcing_fft', value=0.)

        if params.forcing.nkmax_forcing < params.forcing.nkmin_forcing:
            raise ValueError('params.forcing.nkmax_forcing < \n'
                             'params.forcing.nkmin_forcing')
        self.kmax_forcing = self.oper.deltakh * params.forcing.nkmax_forcing
        self.kmin_forcing = self.oper.deltakh * params.forcing.nkmin_forcing

        self.forcing_rate = params.forcing.forcing_rate

        if params.forcing.key_forced is not None:
            self.key_forced = params.forcing.key_forced
        else:
            self.key_forced = self._key_forced_default

        i = 0
        while 2 * params.forcing.nkmax_forcing > 2**i:
            i += 1
        n = 2**i

        self._check_forcing_shape([n], sim.oper.shapeX_seq)

        if mpi.rank == 0:
            params_coarse = deepcopy(params)
            params_coarse.oper.nx = n
            params_coarse.oper.ny = n
            params_coarse.oper.type_fft = 'sequential'
            # FIXME: Workaround for incorrect forcing
            params_coarse.oper.coef_dealiasing = 1.

            self.oper_coarse = sim.oper.__class__(
                SEQUENTIAL=True,
                params=params_coarse,
                goal_to_print='coarse resolution for forcing')
            self.shapeK_loc_coarse = self.oper_coarse.shapeK_loc

            self.COND_NO_F = self._compute_cond_no_forcing()

            self.nb_forced_modes = (self.COND_NO_F.size -
                                    np.array(self.COND_NO_F,
                                             dtype=np.int32).sum())
            self.ind_forcing = np.logical_not(
                self.COND_NO_F).flatten().nonzero()[0]

            self.fstate_coarse = sim.state.__class__(
                sim, oper=self.oper_coarse)

        else:
            self.shapeK_loc_coarse = None

        if mpi.nb_proc > 1:
            self.shapeK_loc_coarse = mpi.comm.bcast(
                self.shapeK_loc_coarse, root=0)

        # if params.forcing.type_forcing == 'WAVES':
        #     self.compute = self.compute_forcing_waves
        #     if mpi.rank == 0:
        #         eta_rms_max = 0.1
        #         self.eta_cond = eta_rms_max / np.sqrt(self.nb_forced_modes)
        #         print '    eta_cond =', self.eta_cond

    def _compute_cond_no_forcing(self):
        return np.logical_or(
            self.oper_coarse.KK > self.kmax_forcing,
            self.oper_coarse.KK < self.kmin_forcing)

    def put_forcingc_in_forcing(self):
        """Copy data from forcingc_fft into forcing_fft."""
        nKyc = self.shapeK_loc_coarse[0]
        nKxc = self.shapeK_loc_coarse[1]
        nb_keys = self.forcing_fft.nvar

        ar3Df = self.forcing_fft
        if mpi.rank == 0:
            # ar3Dfc = self.forcingc_fft
            ar3Dfc = self.fstate_coarse.state_spect

        if mpi.nb_proc > 1:
            if not self.oper.is_transposed:
                raise NotImplementedError()
            nKy = self.oper.shapeK_seq[1]

            for ikey in range(nb_keys):
                if mpi.rank == 0:
                    fck_fft = ar3Dfc[ikey].transpose()

                for iKxc in range(nKxc):
                    kx = self.oper.deltakx * iKxc
                    rank_iKx, iKxloc, iKyloc = (
                        self.oper.where_is_wavenumber(kx, 0.))

                    if mpi.rank == 0:
                        fc1D = fck_fft[iKxc]

                    if rank_iKx != 0:
                        # message fc1D
                        if mpi.rank == rank_iKx:
                            fc1D = np.empty([nKyc], dtype=np.complex128)
                        if mpi.rank == 0 or mpi.rank == rank_iKx:
                            fc1D = np.ascontiguousarray(fc1D)
                        if mpi.rank == 0:
                            mpi.comm.Send([fc1D, mpi.MPI.COMPLEX],
                                          dest=rank_iKx, tag=iKxc)
                        elif mpi.rank == rank_iKx:
                            mpi.comm.Recv([fc1D, mpi.MPI.COMPLEX],
                                          source=0, tag=iKxc)
                    if mpi.rank == rank_iKx:
                        # copy
                        for iKyc in range(nKyc):
                            if iKyc <= nKyc / 2.:
                                iKy = iKyc
                            else:
                                kynodim = iKyc - nKyc
                                iKy = kynodim + nKy
                            ar3Df[ikey, iKxloc, iKy] = fc1D[iKyc]

        else:
            nKy = self.oper.shapeK_seq[0]

            for ikey in range(nb_keys):
                for iKyc in range(nKyc):
                    if iKyc <= nKyc / 2.:
                        iKy = iKyc
                    else:
                        kynodim = iKyc - nKyc
                        iKy = kynodim + nKy
                    for iKxc in range(nKxc):
                        ar3Df[ikey, iKy, iKxc] = ar3Dfc[ikey, iKyc, iKxc]

    def verify_injection_rate(self):
        """Verify injection rate."""
        Fa_fft = self.forcing_fft.get_var(self.key_forced)
        a_fft = self.sim.state.state_spect.get_var(self.key_forced)

        PZ_forcing1 = abs(Fa_fft)**2 / 2 * self.sim.time_stepping.deltat
        PZ_forcing2 = np.real(
            Fa_fft.conj() * a_fft +
            Fa_fft * a_fft.conj()) / 2.
        PZ_forcing1 = self.oper.sum_wavenumbers(PZ_forcing1)
        PZ_forcing2 = self.oper.sum_wavenumbers(PZ_forcing2)
        if mpi.rank == 0:
            print('PZ_f = {0:9.4e} ; PZ_f2 = {1:9.4e};'.format(
                PZ_forcing1 + PZ_forcing2,
                PZ_forcing2))


class InScriptForcingPseudoSpectral(SpecificForcingPseudoSpectral):
    """Forcing maker for forcing defined by the user in the launching script

    .. inheritance-diagram:: InScriptForcingPseudoSpectral

    """
    tag = 'in_script'

    def compute(self):
        """compute a forcing normalize with a 2nd degree eq."""

        if mpi.rank == 0:
            Fa_fft = self.compute_forcingc_fft_each_time()
            kwargs = {self.key_forced: Fa_fft}
            self.fstate_coarse.init_statespect_from(**kwargs)

        self.put_forcingc_in_forcing()

    def compute_forcingc_fft_each_time(self):
        """Compute the coarse forcing in Fourier space"""
        return self.oper_coarse.fft(self.compute_forcingc_each_time())

    def compute_forcingc_each_time(self):
        """Compute the coarse forcing in real space"""
        return self.oper_coarse.create_arrayX_random()

    def monkeypatch_compute_forcingc_fft_each_time(self, func):
        """Replace the method by a user-defined method"""
        self.compute_forcingc_fft_each_time = types.MethodType(func, self)

    def monkeypatch_compute_forcingc_each_time(self, func):
        """Replace the method by a user-defined method"""
        self.compute_forcingc_each_time = types.MethodType(func, self)


class Proportional(SpecificForcingPseudoSpectral):
    """Specific forcing proportional to the forced variable

    .. inheritance-diagram:: Proportional

    """

    tag = 'proportional'

    def compute(self):
        """compute a forcing normalize with a 2nd degree eq."""

        try:
            a_fft = self.sim.state.state_spect.get_var(self.key_forced)
        except ValueError:
            a_fft = self.sim.state.get_var(self.key_forced)

        a_fft = self.oper.coarse_seq_from_fft_loc(
            a_fft, self.shapeK_loc_coarse)

        if mpi.rank == 0:
            Fa_fft = self.forcingc_raw_each_time(a_fft)
            kwargs = {self.key_forced: Fa_fft}
            self.fstate_coarse.init_statespect_from(**kwargs)

        self.put_forcingc_in_forcing()

    def forcingc_raw_each_time(self, vc_fft):
        """Modify the array fvc_fft to fixe the injection rate.

        varc : ndarray
            a variable at the coarse resolution.

        To be called only with proc 0.
        """
        fvc_fft = vc_fft.copy()
        fvc_fft[self.COND_NO_F] = 0.

        Z_fft = abs(fvc_fft)**2 / 2.

        # # possibly "kill" the largest mode
        # nb_kill = 0
        # for ik in xrange(nb_kill):
        #     imax = Z_fft.argmax()
        #     Z_fft.flat[imax] = 0.
        #     fvc_fft.flat[imax] = 0.

        # # possibly add randomness: random kill!
        # nb_kill = self.nb_forced_modes-10
        # ind_kill = random.sample(self.ind_forcing,nb_kill)
        # for ik in ind_kill:
        #     Z_fft.flat[ik] = 0.
        #     fvc_fft.flat[ik] = 0.

        Z = self.oper_coarse.sum_wavenumbers(Z_fft)
        deltat = self.sim.time_stepping.deltat
        alpha = (np.sqrt(1 + deltat * self.forcing_rate / Z) - 1.) / deltat
        fvc_fft = alpha * fvc_fft

        return fvc_fft


class NormalizedForcing(SpecificForcingPseudoSpectral):
    """Specific forcing normalized to keep constant injection

    .. inheritance-diagram:: NormalizedForcing

    """
    tag = 'normalized_forcing'

    @classmethod
    def _complete_params_with_default(cls, params):
        """This static method is used to complete the *params* container.
        """
        super(NormalizedForcing, cls)._complete_params_with_default(params)
        params.forcing._set_child(
            cls.tag,
            {'type_normalize': '2nd_degree_eq'})

    def compute(self):
        """compute a forcing normalize with a 2nd degree eq."""

        try:
            a_fft = self.sim.state.state_spect.get_var(self.key_forced)
        except ValueError:
            a_fft = self.sim.state.get_var(self.key_forced)

        try:
            a_fft = self.oper.coarse_seq_from_fft_loc(
                a_fft, self.shapeK_loc_coarse)
        except IndexError:
            raise ValueError(
                'rank={}, shapeK_loc(coarse)={}, shapeK_loc={}'.format(
                    self.oper.rank, self.shapeK_loc_coarse,
                    self.oper.shapeK_loc))

        if mpi.rank == 0:
            Fa_fft = self.forcingc_raw_each_time(a_fft)
            Fa_fft = self.normalize_forcingc(Fa_fft, a_fft)
            kwargs = {self.key_forced: Fa_fft}
            self.fstate_coarse.init_statespect_from(**kwargs)

        self.put_forcingc_in_forcing()

        # # verification
        # self.verify_injection_rate()

    def normalize_forcingc(self, fvc_fft, vc_fft):
        """Normalize the coarse forcing"""

        type_normalize = self.params.forcing[self.tag].type_normalize

        if type_normalize == '2nd_degree_eq':
            return self.normalize_forcingc_2nd_degree_eq(fvc_fft, vc_fft)
        elif type_normalize == 'particular_k':
            return self.normalize_forcingc_part_k(fvc_fft, vc_fft)
        else:
            ValueError('Bad value for parameter forcing.type_normalize:',
                       type_normalize)

    def normalize_forcingc_part_k(self, fvc_fft, vc_fft):
        """Modify the array fvc_fft to fixe the injection rate.

        To be called only with proc 0.

        Parameters
        ----------

        fvc_fft : ndarray
            The non-normalized forcing at the coarse resolution.

        vc_fft : ndarray
            The forced variable at the coarse resolution.

        """
        oper_c = self.oper_coarse

        oper_c.project_fft_on_realX(fvc_fft)
        # fvc_fft[self.COND_NO_F] = 0.

        P_forcing2 = np.real(
            fvc_fft.conj() * vc_fft +
            fvc_fft * vc_fft.conj()) / 2.
        P_forcing2 = oper_c.sum_wavenumbers(P_forcing2)

        # we choice randomly a "particular" wavenumber
        # in the forced space
        KX_f = oper_c.KX[~self.COND_NO_F].flatten()
        KY_f = oper_c.KY[~self.COND_NO_F].flatten()
        nb_wn_f = len(KX_f)

        ipart = np.random.random_integers(0, nb_wn_f - 1)
        kx_part = KX_f[ipart]
        ky_part = KY_f[ipart]
        ikx_part = abs(oper_c.kx_loc - kx_part).argmin()
        iky_part = abs(oper_c.ky_loc - ky_part).argmin()

        ik0_part = iky_part
        ik1_part = ikx_part

        P_forcing2_part = np.real(
            fvc_fft[ik0_part, ik1_part].conj() *
            vc_fft[ik0_part, ik1_part] +
            fvc_fft[ik0_part, ik1_part] *
            vc_fft[ik0_part, ik1_part].conj())

        if ikx_part == 0:
            P_forcing2_part = P_forcing2_part / 2.
        P_forcing2_other = P_forcing2 - P_forcing2_part
        fvc_fft[ik0_part, ik1_part] = \
            -P_forcing2_other / vc_fft[ik0_part, ik1_part].real

        if ikx_part != 0:
            fvc_fft[ik0_part, ik1_part] = fvc_fft[ik0_part, ik1_part] / 2.

        oper_c.project_fft_on_realX(fvc_fft)

        # normalisation to obtain the wanted total forcing rate
        PZ_nonorm = (oper_c.sum_wavenumbers(abs(fvc_fft)**2) *
                     self.sim.time_stepping.deltat / 2
                     )
        fvc_fft = fvc_fft * np.sqrt(float(self.forcing_rate) / PZ_nonorm)

        return fvc_fft

    def normalize_forcingc_2nd_degree_eq(self, fvc_fft, vc_fft):
        """Modify the array fvc_fft to fixe the injection rate.

        To be called only with proc 0.

        Parameters
        ----------

        fvc_fft : ndarray
            The non-normalized forcing at the coarse resolution.

        vc_fft : ndarray
            The forced variable at the coarse resolution.
        """
        oper_c = self.oper_coarse

        deltat = self.sim.time_stepping.deltat

        a = deltat / 2 * oper_c.sum_wavenumbers(abs(fvc_fft)**2)

        b = oper_c.sum_wavenumbers(
            (vc_fft.conj() * fvc_fft).real)

        c = -self.forcing_rate

        Delta = b**2 - 4 * a * c
        alpha = (np.sqrt(Delta) - b) / (2 * a)

        fvc_fft = alpha * fvc_fft

        return fvc_fft

    def coef_normalization_from_abc(self, a, b, c):
        """."""
        Delta = b**2 - 4 * a * c
        alpha = (np.sqrt(Delta) - b) / (2 * a)
        return alpha


class RandomSimplePseudoSpectral(NormalizedForcing):
    """Random normalized forcing

    .. inheritance-diagram:: RandomSimplePseudoSpectral
    """
    tag = 'random'

    def compute_forcingc_raw(self):
        """Random coarse forcing.

        To be called only with proc 0.
        """
        F_fft = self.oper_coarse.create_arrayK_random()
        # fftwpy/easypyfft returns F_fft
        F_fft = self.oper_coarse.project_fft_on_realX(F_fft)
        F_fft[self.COND_NO_F] = 0.
        return F_fft

    def forcingc_raw_each_time(self, a_fft):
        return self.compute_forcingc_raw()


class TimeCorrelatedRandomPseudoSpectral(RandomSimplePseudoSpectral):
    """Time correlated random normalized forcing

    .. inheritance-diagram:: TimeCorrelatedRandomPseudoSpectral
    """
    tag = 'tcrandom'

    @classmethod
    def _complete_params_with_default(cls, params):
        """This static method is used to complete the *params* container.
        """
        super(TimeCorrelatedRandomPseudoSpectral,
              cls)._complete_params_with_default(params)
        params.forcing[cls.tag]._set_attrib(
            'time_correlation', 'based_on_forcing_rate')

    def __init__(self, sim):

        super(TimeCorrelatedRandomPseudoSpectral, self).__init__(sim)

        if mpi.rank == 0:
            self.F0 = self.compute_forcingc_raw()
            self.F1 = self.compute_forcingc_raw()

            time_correlation = self.params.forcing[self.tag].time_correlation
            if time_correlation == 'based_on_forcing_rate':
                self.period_change_F0F1 = self.forcing_rate**(-1. / 3)
            else:
                self.period_change_F0F1 = time_correlation
            self.t_last_change = self.sim.time_stepping.t

    def forcingc_raw_each_time(self, a_fft):
        """Return a coarse forcing as a linear combination of 2 random arrays

        Compute the new random coarse forcing arrays when necessary.

        """
        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_change >= self.period_change_F0F1:
            self.t_last_change = tsim
            self.F0 = self.F1
            del(self.F1)
            self.F1 = self.compute_forcingc_raw()

        F_fft = self.forcingc_from_F0F1()
        return F_fft

    def forcingc_from_F0F1(self):
        """Return a coarse forcing as a linear combination of 2 random arrays

        """
        tsim = self.sim.time_stepping.t
        deltat = self.period_change_F0F1
        omega = np.pi / deltat

        deltaF = self.F1 - self.F0

        F_fft = self.F1 - 0.5 * (
            np.cos((tsim - self.t_last_change) * omega) + 1) * deltaF

        return F_fft


class TimeCorrelatedRandomPseudoSpectralAnisotropic(
        TimeCorrelatedRandomPseudoSpectral):
    """Random normalized anisotropic forcing.

    .. inheritance-diagram:: TimeCorrelatedRandomPseudoSpectralAnisotropic

    """
    tag = 'tcrandom_anisotropic'

    @classmethod
    def _complete_params_with_default(cls, params):
        """This static method is used to complete the *params* container.
        """
        super(TimeCorrelatedRandomPseudoSpectral,
              cls)._complete_params_with_default(params)

        params.forcing[cls.tag]._set_attribs({
            'angle': '45',
            'time_correlation': 'based_on_forcing_rate'})

    def _compute_cond_no_forcing(self):
        """Computes condition no forcing of the anisotropic case.
        """
        angle = radians(float(self.params.forcing[self.tag].angle))

        kxmin_forcing = np.sin(angle) * self.kmin_forcing
        kxmax_forcing = np.sin(angle) * self.kmax_forcing

        kymin_forcing = np.cos(angle) * self.kmin_forcing
        kymax_forcing = np.cos(angle) * self.kmax_forcing

        if kxmax_forcing - kxmin_forcing < self.oper.deltakx or \
           kymax_forcing - kymin_forcing < self.oper.deltaky:
            raise ValueError('No forcing modes in one direction.')

        COND_NO_F_KX = np.logical_or(
            self.oper_coarse.KX > kxmax_forcing,
            self.oper_coarse.KX < kxmin_forcing)

        COND_NO_F_KY = np.logical_or(
            self.oper_coarse.KY > kymax_forcing,
            self.oper_coarse.KY < kymin_forcing)

        return np.logical_or(COND_NO_F_KX, COND_NO_F_KY)
