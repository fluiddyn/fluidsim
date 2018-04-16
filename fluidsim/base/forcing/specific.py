"""Forcing schemes (:mod:`fluidsim.base.forcing.specific`)
================================================================

Provides:

.. autoclass:: SpecificForcing
   :members:
   :private-members:

.. autoclass:: SpecificForcingPseudoSpectralSimple
   :members:
   :private-members:

.. autoclass:: InScriptForcingPseudoSpectral
   :members:
   :private-members:

.. autoclass:: SpecificForcingPseudoSpectral
   :members:
   :private-members:

.. autoclass:: InScriptForcingPseudoSpectralCoarse
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

from builtins import object

from copy import deepcopy
from math import radians, pi
import types
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from fluiddyn.util import mpi
from fluiddyn.calcul.easypyfft import fftw_grid_size
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


class SpecificForcingPseudoSpectralSimple(SpecificForcing):
    """Specific forcing for pseudo-spectra solvers"""

    tag = 'pseudo_spectral'

    def __init__(self, sim):
        super(SpecificForcingPseudoSpectralSimple, self).__init__(sim)
        self.fstate = sim.state.__class__(
            sim, oper=self.sim.oper)
        self.forcing_fft = self.fstate.state_spect


class InScriptForcingPseudoSpectral(SpecificForcingPseudoSpectralSimple):
    """Forcing maker for forcing defined by the user in the launching script

    .. inheritance-diagram:: InScriptForcingPseudoSpectral

    """
    tag = 'in_script'

    def compute(self):
        """compute a forcing normalize with a 2nd degree eq."""
        obj = self.compute_forcing_fft_each_time()
        if isinstance(obj, dict):
            kwargs = obj
        else:
            kwargs = {self.sim.params.forcing.key_forced: obj}
        self.fstate.init_statespect_from(**kwargs)

    def compute_forcing_fft_each_time(self):
        """Compute the coarse forcing in Fourier space"""
        obj = self.compute_forcing_each_time()
        if isinstance(obj, dict):
            kwargs = {key: self.sim.oper.fft(value)
                      for key, value in obj.items()}
        else:
            kwargs = {self.sim.params.forcing.key_forced:
                      self.sim.oper.fft(obj)}
        return kwargs

    def compute_forcing_each_time(self):
        """Compute the coarse forcing in real space"""
        return self.sim.oper.create_arrayX(value=0)

    def monkeypatch_compute_forcing_fft_each_time(self, func):
        """Replace the method by a user-defined method"""
        self.compute_forcing_fft_each_time = types.MethodType(func, self)

    def monkeypatch_compute_forcing_each_time(self, func):
        """Replace the method by a user-defined method"""
        self.compute_forcing_each_time = types.MethodType(func, self)


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

        """
        if any(np.greater(shape_forcing, shape)):
            raise NotImplementedError(
                'The resolution is too small for the required forcing: '
                'any{} < {}'.format(shape, shape_forcing))

    def __init__(self, sim):

        super(SpecificForcingPseudoSpectral, self).__init__(sim)

        params = sim.params

        self.forcing_fft = SetOfVariables(
            like=sim.state.state_spect, info='forcing_fft', value=0.)

        if params.forcing.nkmax_forcing < params.forcing.nkmin_forcing:
            raise ValueError('params.forcing.nkmax_forcing < \n'
                             'params.forcing.nkmin_forcing')
        self.kmax_forcing = self.oper.deltak * params.forcing.nkmax_forcing
        self.kmin_forcing = self.oper.deltak * params.forcing.nkmin_forcing

        self.forcing_rate = params.forcing.forcing_rate

        if params.forcing.key_forced is not None:
            self.key_forced = params.forcing.key_forced
        else:
            self.key_forced = self._key_forced_default

        try:
            n = 2 * fftw_grid_size(params.forcing.nkmax_forcing)
        except ImportError:
            warn('To use smaller forcing arrays: pip install pulp')
            i = 0
            while 2 * params.forcing.nkmax_forcing > 2**i:
                i += 1
            n = 2**i

        self._check_forcing_shape([n], sim.oper.shapeX_seq)

        try:
            angle = radians(float(self.params.forcing[self.tag].angle))
        except AttributeError:
            pass
        else:
            self.kxmax_forcing = np.sin(angle) * self.kmax_forcing
            self.kymax_forcing = np.cos(angle) * self.kmax_forcing

        if mpi.rank == 0:
            params_coarse = deepcopy(params)

            params_coarse.oper.nx = n
            # The 2 * deltakx aims to give some gap between the kxmax and
            # the boundary of the oper_coarse.
            try:
                params_coarse.oper.nx = fftw_grid_size(int(
                    (self.kxmax_forcing + 2 * self.oper.deltakx) * (
                        params.oper.Lx / pi)))
            except AttributeError:
                pass
            try:
                params_coarse.oper.ny = n
                try:
                    params_coarse.oper.ny = fftw_grid_size(int(
                        (self.kymax_forcing + 2 * self.oper.deltaky) * (
                            params.oper.Ly / pi)))
                except AttributeError:
                    pass
            except AttributeError:
                pass
            try:
                params_coarse.oper.nz = n
            except AttributeError:
                pass
            params_coarse.oper.type_fft = 'sequential'
            # FIXME: Workaround for incorrect forcing
            params_coarse.oper.coef_dealiasing = 1.

            self.oper_coarse = sim.oper.__class__(
                params=params_coarse)
            self.shapeK_loc_coarse = self.oper_coarse.shapeK_loc
            print('self.shapeK_loc_coarse', self.shapeK_loc_coarse)
            self.COND_NO_F = self._compute_cond_no_forcing()

            self.nb_forced_modes = (self.COND_NO_F.size -
                                    np.array(self.COND_NO_F,
                                             dtype=np.int32).sum())
            if not self.nb_forced_modes:
                raise ValueError('0 modes forced.')

            try:
                hasattr(self, 'plot_forcing_region')
            except NotImplementedError:
                pass
            else:
                mpi.printby0('To plot the forcing modes, you can use:\n'
                              'sim.forcing.forcing_maker.plot_forcing_region()')

            self.ind_forcing = np.logical_not(
                self.COND_NO_F).flatten().nonzero()[0]

            self.fstate_coarse = sim.state.__class__(
                sim, oper=self.oper_coarse)
        else:
            self.shapeK_loc_coarse = None

        if mpi.nb_proc > 1:
            self.shapeK_loc_coarse = mpi.comm.bcast(
                self.shapeK_loc_coarse, root=0)

    def _compute_cond_no_forcing(self):
        if hasattr(self.oper_coarse, 'K'):
            K = self.oper_coarse.K
        else:
            K = np.sqrt(self.oper_coarse.K2)

        return np.logical_or(K > self.kmax_forcing, K < self.kmin_forcing)

    def put_forcingc_in_forcing(self):
        """Copy data from self.fstate_coarse.state_spect into forcing_fft."""
        if mpi.rank == 0:
            state_spect = self.fstate_coarse.state_spect
            oper_coarse = self.oper_coarse
        else:
            state_spect = None
            oper_coarse = None

        self.oper.put_coarse_array_in_array_fft(
            state_spect, self.forcing_fft, oper_coarse,
            self.shapeK_loc_coarse)

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


class InScriptForcingPseudoSpectralCoarse(SpecificForcingPseudoSpectral):
    """Forcing maker for forcing defined by the user in the launching script

    .. inheritance-diagram:: InScriptForcingPseudoSpectralCoarse

    """
    tag = 'in_script_coarse'

    def compute(self):
        """compute a forcing normalize with a 2nd degree eq."""

        if mpi.rank == 0:
            obj = self.compute_forcingc_fft_each_time()
            if isinstance(obj, dict):
                kwargs = obj
            else:
                kwargs = {self.key_forced: obj}
            self.fstate_coarse.init_statespect_from(**kwargs)

        self.put_forcingc_in_forcing()

    def compute_forcingc_fft_each_time(self):
        """Compute the coarse forcing in Fourier space"""
        return self.oper_coarse.fft(self.compute_forcingc_each_time())

    def compute_forcingc_each_time(self):
        """Compute the coarse forcing in real space"""
        return self.oper_coarse.create_arrayX(value=0)

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
    tag = 'normalized'

    @classmethod
    def _complete_params_with_default(cls, params):
        """This static method is used to complete the *params* container.
        """
        super(NormalizedForcing, cls)._complete_params_with_default(params)
        try:
            params.forcing.normalized
        except AttributeError:
            params.forcing._set_child(
                'normalized', {'type': '2nd_degree_eq', 'which_root': 'first'})

    def __init__(self, sim):
        super(NormalizedForcing, self).__init__(sim)

        if self.params.forcing.normalized.type == 'particular_k':
            raise NotImplementedError

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

        type_normalize = self.params.forcing.normalized.type

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

        # warning : this is 2d specific!

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

        alpha = self.coef_normalization_from_abc(a, b, c)

        fvc_fft = alpha * fvc_fft

        return fvc_fft

    def coef_normalization_from_abc(self, a, b, c):
        """Compute the roots of a quadratic equation

        Compute the roots given the coefficients `a`,`b` and `c`.
        Then, select one of the roots based on a criteria and return it.

        Notes
        -----
        Set params.forcing.normalized.which_root to choose the root with:
            
        - `minabs` : minimum absolute value
        - `first` : root with positive sign before discriminant
        - `second` : root with negative sign before discriminant
        - `positive` : positive root

        """
        try:
            alpha1, alpha2 = np.roots([a, b, c])
        except ValueError:
            return 0.

        which_root = self.params.forcing.normalized.which_root

        if which_root == 'minabs':
            if abs(alpha2) < abs(alpha1):
                return alpha2
            else:
                return alpha1
        elif which_root == 'first':
            return alpha1
        elif which_root == 'second':
            return alpha2
        elif which_root == 'positive':
            if alpha2 > 0.:
                return alpha2
            else:
                return alpha1
        else:
            raise ValueError(
                'Not sure how to choose which root to normalize forcing with.')


class RandomSimplePseudoSpectral(NormalizedForcing):
    """Random normalized forcing

    .. inheritance-diagram:: RandomSimplePseudoSpectral
    """
    tag = 'random'

    @classmethod
    def _complete_params_with_default(cls, params):
        """This static method is used to complete the *params* container.
        """
        super(RandomSimplePseudoSpectral,
              cls)._complete_params_with_default(params)

        try:
            params.forcing.random
        except AttributeError:
            params.forcing._set_child(
                'random', {'only_positive': False})

    def __init__(self, sim):
        
        super(RandomSimplePseudoSpectral, self).__init__(sim)

        if self.params.forcing.random.only_positive:
            self._min_val = None
        else:
            self._min_val = -1

    def compute_forcingc_raw(self):
        """Random coarse forcing.

        To be called only with proc 0.
        """
        f_fft = self.oper_coarse.create_arrayK_random(min_val=self._min_val)
        # fftwpy/easypyfft returns f_fft
        f_fft = self.oper_coarse.project_fft_on_realX(f_fft)
        f_fft[self.COND_NO_F] = 0.
        return f_fft

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

        try:
            params.forcing.tcrandom
        except AttributeError:
            params.forcing._set_child(
                'tcrandom', {'time_correlation': 'based_on_forcing_rate'})

    def __init__(self, sim):

        super(TimeCorrelatedRandomPseudoSpectral, self).__init__(sim)

        if mpi.rank == 0:
            self.forcing0 = self.compute_forcingc_raw()
            self.forcing1 = self.compute_forcingc_raw()

            pforcing = self.params.forcing
            try:
                time_correlation = pforcing[self.tag].time_correlation
            except AttributeError:
                time_correlation = pforcing.tcrandom.time_correlation

            if time_correlation == 'based_on_forcing_rate':
                self.period_change_f0f1 = self.forcing_rate**(-1. / 3)
            else:
                self.period_change_f0f1 = time_correlation
            self.t_last_change = self.sim.time_stepping.t

    def forcingc_raw_each_time(self, a_fft):
        """Return a coarse forcing as a linear combination of 2 random arrays

        Compute the new random coarse forcing arrays when necessary.

        """
        tsim = self.sim.time_stepping.t
        if tsim - self.t_last_change >= self.period_change_f0f1:
            self.t_last_change = tsim
            self.forcing0 = self.forcing1
            self.forcing1 = self.compute_forcingc_raw()

        f_fft = self.forcingc_from_f0f1()
        return f_fft

    def forcingc_from_f0f1(self):
        """Return a coarse forcing as a linear combination of 2 random arrays

        """
        tsim = self.sim.time_stepping.t
        deltat = self.period_change_f0f1
        omega = np.pi / deltat

        deltaf = self.forcing1 - self.forcing0

        f_fft = self.forcing1 - 0.5 * (
            np.cos((tsim - self.t_last_change) * omega) + 1) * deltaf

        return f_fft


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

        params.forcing._set_child(
            'tcrandom_anisotropic', {'angle': '45'})

    def _compute_cond_no_forcing(self):
        """Computes condition no forcing of the anisotropic case.
        """
        angle = radians(float(self.params.forcing[self.tag].angle))

        self.kxmin_forcing = np.sin(angle) * self.kmin_forcing
        self.kxmax_forcing = np.sin(angle) * self.kmax_forcing

        self.kymin_forcing = np.cos(angle) * self.kmin_forcing
        self.kymax_forcing = np.cos(angle) * self.kmax_forcing

        if self.kxmax_forcing - self.kxmin_forcing < self.oper.deltakx or \
           self.kymax_forcing - self.kymin_forcing < self.oper.deltaky:
            raise ValueError('No forcing modes in one direction.')

        COND_NO_F_KX = np.logical_or(
            self.oper_coarse.KX > self.kxmax_forcing,
            self.oper_coarse.KX < self.kxmin_forcing)

        COND_NO_F_KY = np.logical_or(
            self.oper_coarse.KY > self.kymax_forcing,
            self.oper_coarse.KY < self.kymin_forcing)

        return np.logical_or(COND_NO_F_KX, COND_NO_F_KY)

    def plot_forcing_region(self):
        """Plots the forcing region"""
        pforcing = self.params.forcing
        oper = self.oper

        kxmin_forcing = self.kxmin_forcing
        kxmax_forcing = self.kxmax_forcing
        kymin_forcing = self.kymin_forcing
        kymax_forcing = self.kymax_forcing

        # Define forcing region
        coord_x = kxmin_forcing
        coord_y = kymin_forcing
        width = kxmax_forcing - kxmin_forcing
        height = kymax_forcing - kymin_forcing

        theta1 = 90.0 - float(pforcing.tcrandom_anisotropic.angle)
        theta2 = 90.0

        KX = self.oper_coarse.KX
        KY = self.oper_coarse.KY

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        title = (
            pforcing.type + '; ' +
            r'$nk_{{min}} = {} \delta k_x$; '.format(pforcing.nkmin_forcing) +
            r'$nk_{{max}} = {} \delta k_z$; '.format(pforcing.nkmax_forcing) +
            r'$\theta = {}^\circ$; '.format(
                pforcing.tcrandom_anisotropic.angle) +
            r'Forced modes = {}'.format(self.nb_forced_modes))

        ax.set_title(title)
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_z$')

        # Parameters figure
        ax.set_xlim([abs(KX).min(), abs(KX).max()])
        ax.set_ylim([abs(KY).min(), abs(KY).max()])

        # Set ticks 10% of the KX.max and KY.max
        factor = 0.1
        sep_x = abs(KX).max() * factor
        sep_y = abs(KY).max() * factor
        nb_deltakx = int(sep_x // self.oper.deltakx)
        nb_deltaky = int(sep_y // self.oper.deltaky)

        if not nb_deltakx:
            nb_deltakx = 1
        if not nb_deltaky:
             nb_deltaky = 1

        xticks = np.arange(
            abs(KX).min(), abs(KX).max(), nb_deltakx * self.oper.deltakx)
        yticks = np.arange(
            abs(KY).min(), abs(KY).max(), nb_deltaky * self.oper.deltaky)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.add_patch(patches.Rectangle(
            xy=(coord_x, coord_y),
            width=width,
            height=height,
            fill=False))

        # width and height arc 50% the length of the axis
        ax.add_patch(patches.Arc(
            xy=(0, 0),
            width=abs(KX).max() * 0.5,
            height=abs(KX).max() * 0.5,
            angle=0, theta1=theta1, theta2=theta2))

        # Plot arc kmin and kmax
        ax.add_patch(patches.Arc(
            xy=(0,0),
            width=2 * self.kmin_forcing,
            height=2 * self.kmin_forcing,
            angle=0, theta1=0, theta2=90.0,
            linestyle='-.'))
        ax.add_patch(patches.Arc(
            xy=(0,0),
            width=2 * self.kmax_forcing,
            height=2 * self.kmax_forcing,
            angle=0, theta1=0, theta2=90.0,
            linestyle='-.'))

        # Plot lines angle & lines forcing region
        ax.plot([0, kxmin_forcing], [0, kymin_forcing], color='k', linewidth=1)
        ax.plot([kxmin_forcing, kxmin_forcing], [0, kymin_forcing],
                'k--', linewidth=0.8)
        ax.plot([kxmax_forcing, kxmax_forcing], [0, kymin_forcing],
                'k--', linewidth=0.8)
        ax.plot([0, kxmin_forcing], [kymin_forcing, kymin_forcing],
                'k--', linewidth=0.8)
        ax.plot([0, kxmin_forcing], [kymax_forcing, kymax_forcing],
                'k--', linewidth=0.8)

        # Plot forced modes in red
        indices_forcing = np.argwhere(self.COND_NO_F == False)
        for i, index in enumerate(indices_forcing):
                ax.plot(KX[0, index[1]], KY[index[0], 0],
                        'ro', label='Forced mode' if i == 0 else "")

        # Location labels 0.8% the length of the axis
        factor = 0.008
        loc_label_y = abs(KY).max() * factor
        loc_label_x = abs(KX).max() * factor

        ax.text(loc_label_x + kxmin_forcing,  loc_label_y, r'$k_{x,min}$')
        ax.text(loc_label_x + kxmax_forcing,  loc_label_y, r'$k_{x,max}$')
        ax.text(loc_label_x, kymin_forcing + loc_label_y, r'$k_{z,min}$')
        ax.text(loc_label_x, kymax_forcing + loc_label_y, r'$k_{z,max}$')

        # Location label angle \theta
        factor_x = 0.015
        factor_y = 0.15
        loc_label_y = abs(KY).max() * factor_y
        loc_label_x = abs(KX).max() * factor_x

        ax.text(loc_label_x, loc_label_y, r'$\theta$')

        ax.grid(linestyle='--', alpha=0.4)
        ax.legend()
