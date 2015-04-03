"""
Time stepping Cython (:mod:`fluidsim.base.time_stepping.pseudo_spect_cy`)
=========================================================================

.. currentmodule:: fluidsim.base.time_stepping.pseudo_spect_cy

Provides:

.. autoclass:: ExactLinearCoefs
   :members:
   :private-members:

.. autoclass:: TimeSteppingPseudoSpectral
   :members:
   :private-members:

"""

cimport numpy as np
import numpy as np
np.import_array()

from time import time, sleep
import datetime
import os
import matplotlib.pyplot as plt
import cython

from libc.math cimport exp

from fluidsim.base.setofvariables import SetOfVariables

from pseudo_spect import ExactLinearCoefs as ExactLinearCoefsPurePython
from pseudo_spect import TimeSteppingPseudoSpectral as \
    TimeSteppingPseudoSpectralPurePython


# we define python and c types for physical and Fourier spaces
DTYPEb = np.uint8
ctypedef np.uint8_t DTYPEb_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t
DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t
DTYPEc = np.complex128
ctypedef np.complex128_t DTYPEc_t

# Basically, you use the _t ones when you need to declare a type
# (e.g. cdef foo_t var, or np.ndarray[foo_t, ndim=...]. Ideally someday
# we won't have to make this distinction, but currently one is a C type
# and the other is a python object representing a numpy type (a dtype),
# and there's currently no way to identify the two without special
# compiler support.
# - Robert Bradshaw


cdef extern from "complex.h":
    np.complex128_t cexp(np.complex128_t z) nogil


class ExactLinearCoefs(ExactLinearCoefsPurePython):
    """Handle the computation of the exact coefficient for the RK4."""

    def __init__(self, time_stepping):
        super(ExactLinearCoefs, self).__init__(time_stepping)

        ndim = self.freq_lin.ndim
        dtype = self.freq_lin.dtype

        if ndim == 2 and dtype == np.float64:
            self.compute = self.compute_ndim2_float64
        else:
            raise NotImplementedError(
                'ndim: {} ; dtype {}'.format(ndim, dtype))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_ndim2_float64(self, double dt):
        cdef Py_ssize_t i0, i1, n0, n1
        cdef np.ndarray[double, ndim=2] exact, exact2, f_lin

        exact = self.exact
        exact2 = self.exact2
        f_lin = self.freq_lin
        n0 = exact.shape[0]
        n1 = exact.shape[1]

        for i0 in xrange(n0):
            for i1 in xrange(n1):
                exact[i0, i1] = exp(-dt*f_lin[i0, i1])
                exact2[i0, i1] = exp(-dt/2*f_lin[i0, i1])
        self.dt_old = dt

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_ndim3_complex128(self, double dt):
        cdef Py_ssize_t i0, i1, ik, nk, n0, n1
        cdef np.ndarray[DTYPEc_t, ndim=3] exact, exact2, f_lin

        nk = self.nk
        n0 = self.n0
        n1 = self.n1
        exact = self.exact
        exact2 = self.exact2
        f_lin = self.freq_lin

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    exact[ik, i0, i1] = cexp(-dt*f_lin[ik, i0, i1])
                    exact2[ik, i0, i1] = cexp(-dt/2*f_lin[ik, i0, i1])

        self.dt_old = dt


class TimeSteppingPseudoSpectral(TimeSteppingPseudoSpectralPurePython):

    def _init_time_scheme(self):

        params_ts = self.params.time_stepping

        if params_ts.type_time_scheme not in ['RK2', 'RK4']:
            raise ValueError('Problem name time_scheme')

        dtype = self.freq_lin.dtype
        if dtype == np.float64:
            str_type = 'float'
        elif dtype == np.complex128:
            str_type = 'complex'
        else:
            raise NotImplementedError('dtype of freq_lin:' + repr(dtype))

        name_function = (
            '_time_step_' + params_ts.type_time_scheme +
            '_state_ndim{}_freqlin_ndim{}_'.format(
                self.sim.state.state_fft.ndim, self.freq_lin.ndim) +
            str_type)

        if not hasattr(self, name_function):
            raise NotImplementedError(
                'The function ' + name_function +
                ' is not implemented.')

        exec('self._time_step_RK = self.' + name_function,
             globals(), locals())

    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _time_step_RK4_state_ndim3_freqlin_ndim2_float(self):
        """Advance in time *sim.state.state_fft* with the Runge-Kutta 4 method.

        See :ref:`the pure python RK4 function <rk4timescheme>` for the
        presentation of the time scheme.

        For this function, the coefficient :math:`\sigma` is real and
        represents the dissipation.

        """
        # cdef DTYPEf_t dt = self.deltat
        cdef double dt = self.deltat

        cdef Py_ssize_t i0, i1, ik, nk, n0, n1

        # cdef np.ndarray[DTYPEf_t, ndim=2] exact, exact2
        # This is strange, if I use DTYPEf_t and complex.h => bug
        cdef np.ndarray[double, ndim=2] exact, exact2

        cdef np.ndarray[DTYPEc_t, ndim=3] datas, datat
        cdef np.ndarray[DTYPEc_t, ndim=3] datatemp, datatemp2

        tendencies_nonlin = self.sim.tendencies_nonlin
        state_fft = self.sim.state.state_fft

        nk = state_fft.shape[0]
        n0 = state_fft.shape[1]
        n1 = state_fft.shape[2]

        exact, exact2 = self.exact_linear_coefs.get_updated_coefs()

        tendencies_fft_1 = tendencies_nonlin()

        # # alternativelly, this
        # state_fft_temp = (self.state_fft + dt/6*tendencies_fft_1)*exact
        # state_fft_np12_approx1 = (
        #     self.state_fft + dt/2*tendencies_fft_1)*exact2
        # # or this (slightly faster...)

        datas = state_fft
        datat = tendencies_fft_1

        state_fft_temp = SetOfVariables(like=state_fft)
        datatemp = state_fft_temp

        state_fft_np12_approx1 = SetOfVariables(like=state_fft)
        datatemp2 = state_fft_np12_approx1

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/6*datat[ik, i0, i1])*exact[i0, i1]
                    datatemp2[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])*exact2[i0, i1]

        del(tendencies_fft_1)
        tendencies_fft_2 = tendencies_nonlin(state_fft_np12_approx1)
        del(state_fft_np12_approx1)

        # # alternativelly, this
        # state_fft_temp += dt/3*exact2*tendencies_fft_2
        # state_fft_np12_approx2 = (exact2*self.state_fft
        #                           + dt/2*tendencies_fft_2)
        # # or this (slightly faster...)

        datat = tendencies_fft_2

        state_fft_np12_approx2 = SetOfVariables(like=state_fft)
        datatemp2 = state_fft_np12_approx2

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact2[i0, i1]*datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])

        del(tendencies_fft_2)
        tendencies_fft_3 = tendencies_nonlin(state_fft_np12_approx2)
        del(state_fft_np12_approx2)

        # # alternativelly, this
        # state_fft_temp += dt/3*exact2*tendencies_fft_3
        # state_fft_np1_approx = (exact*self.state_fft
        #                         + dt*exact2*tendencies_fft_3)
        # # or this (slightly faster...)

        datat = tendencies_fft_3

        state_fft_np1_approx = SetOfVariables(like=state_fft)
        datatemp2 = state_fft_np1_approx

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact[i0, i1]*datas[ik, i0, i1] +
                        dt*exact2[i0, i1]*datat[ik, i0, i1])

        del(tendencies_fft_3)
        tendencies_fft_4 = tendencies_nonlin(state_fft_np1_approx)
        del(state_fft_np1_approx)

        # # alternativelly, this
        # self.state_fft = state_fft_temp + dt/6*tendencies_fft_4
        # # or this (slightly faster... may be not...)

        datat = tendencies_fft_4

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datas[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/6*datat[ik, i0, i1])

    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _time_step_RK4_state_ndim3_freqlin_ndim3_float(self):
        """Advance in time *sim.state.state_fft* with the Runge-Kutta 4 method.

        See :ref:`the pure python RK4 function <rk4timescheme>` for the
        presentation of the time scheme.

        For this function, the coefficient :math:`\sigma` is complex.

        """
        cdef double dt = self.deltat
        cdef Py_ssize_t i0, i1, ik, nk, n0, n1
        cdef np.ndarray[double, ndim=3] exact, exact2
        cdef np.ndarray[DTYPEc_t, ndim=3] datas, datat
        cdef np.ndarray[DTYPEc_t, ndim=3] datatemp, datatemp2

        tendencies_nonlin = self.sim.tendencies_nonlin

        state_fft = self.sim.state.state_fft
        datas = state_fft
        nk = datas.shape[0]
        n0 = datas.shape[1]
        n1 = datas.shape[2]

        exact, exact2 = self.exact_linear_coefs.get_updated_coefs()

        tendencies_fft_1 = tendencies_nonlin()

        # # alternativelly, this
        # state_fft_temp = (self.state_fft + dt/6*tendencies_fft_1)*exact
        # state_fft_np12_approx1 = (self.state_fft
        #                           + dt/2*tendencies_fft_1)*exact2
        # # or this (slightly faster...)

        datat = tendencies_fft_1

        state_fft_temp = SetOfVariables(like=state_fft)
        datatemp = state_fft_temp

        state_fft_np12_approx1 = SetOfVariables(like=state_fft)
        datatemp2 = state_fft_np12_approx1

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/6*datat[ik, i0, i1])*exact[ik, i0, i1]
                    datatemp2[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])*exact2[ik, i0, i1]

        del(tendencies_fft_1)
        tendencies_fft_2 = tendencies_nonlin(state_fft_np12_approx1)
        del(state_fft_np12_approx1)

        # # alternativelly, this
        # state_fft_temp += dt/3*exact2*tendencies_fft_2
        # state_fft_np12_approx2 = (exact2*self.state_fft
        #                           + dt/2*tendencies_fft_2)
        # # or this (slightly faster...)

        datat = tendencies_fft_2

        state_fft_np12_approx2 = SetOfVariables(like=state_fft)
        datatemp2 = state_fft_np12_approx2

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[ik, i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact2[ik, i0, i1]*datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])

        del(tendencies_fft_2)
        tendencies_fft_3 = tendencies_nonlin(state_fft_np12_approx2)
        del(state_fft_np12_approx2)

        # # alternativelly, this
        # state_fft_temp += dt/3*exact2*tendencies_fft_3
        # state_fft_np1_approx = (exact*self.state_fft
        #                         + dt*exact2*tendencies_fft_3)
        # # or this (slightly faster...)

        datat = tendencies_fft_3

        state_fft_np1_approx = SetOfVariables(like=state_fft)
        datatemp2 = state_fft_np1_approx

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[ik, i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact[ik, i0, i1]*datas[ik, i0, i1] +
                        dt*exact2[ik, i0, i1]*datat[ik, i0, i1])

        del(tendencies_fft_3)
        tendencies_fft_4 = tendencies_nonlin(state_fft_np1_approx)
        del(state_fft_np1_approx)

        # # alternativelly, this
        # self.state_fft = state_fft_temp + dt/6*tendencies_fft_4
        # # or this (slightly faster... may be not...)

        datat = tendencies_fft_4

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datas[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/6*datat[ik, i0, i1])



    def _time_step_RK2_state_ndim3_freqlin_ndim3_complex(self):
        raise NotImplementedError

    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _time_step_RK4_state_ndim3_freqlin_ndim3_complex(self):
        """Advance in time *sim.state.state_fft* with the Runge-Kutta 4 method.

        See :ref:`the pure python RK4 function <rk4timescheme>` for the
        presentation of the time scheme.

        For this function, the coefficient :math:`\sigma` is complex.

        """
        cdef double dt = self.deltat
        cdef Py_ssize_t i0, i1, ik, nk, n0, n1
        cdef np.ndarray[DTYPEc_t, ndim=3] exact, exact2
        cdef np.ndarray[DTYPEc_t, ndim=3] datas, datat
        cdef np.ndarray[DTYPEc_t, ndim=3] datatemp, datatemp2

        tendencies_nonlin = self.sim.tendencies_nonlin

        state_fft = self.sim.state.state_fft
        datas = state_fft
        nk = datas.shape[0]
        n0 = datas.shape[1]
        n1 = datas.shape[2]

        exact, exact2 = self.exact_linear_coefs.get_updated_coefs()

        tendencies_fft_1 = tendencies_nonlin()

        # # alternativelly, this
        # state_fft_temp = (self.state_fft + dt/6*tendencies_fft_1)*exact
        # state_fft_np12_approx1 = (self.state_fft
        #                           + dt/2*tendencies_fft_1)*exact2
        # # or this (slightly faster...)

        datat = tendencies_fft_1

        state_fft_temp = SetOfVariables(like=state_fft)
        datatemp = state_fft_temp

        state_fft_np12_approx1 = SetOfVariables(like=state_fft)
        datatemp2 = state_fft_np12_approx1

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/6*datat[ik, i0, i1])*exact[ik, i0, i1]
                    datatemp2[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])*exact2[ik, i0, i1]

        del(tendencies_fft_1)
        tendencies_fft_2 = tendencies_nonlin(state_fft_np12_approx1)
        del(state_fft_np12_approx1)

        # # alternativelly, this
        # state_fft_temp += dt/3*exact2*tendencies_fft_2
        # state_fft_np12_approx2 = (exact2*self.state_fft
        #                           + dt/2*tendencies_fft_2)
        # # or this (slightly faster...)

        datat = tendencies_fft_2

        state_fft_np12_approx2 = SetOfVariables(like=state_fft)
        datatemp2 = state_fft_np12_approx2

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[ik, i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact2[ik, i0, i1]*datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])

        del(tendencies_fft_2)
        tendencies_fft_3 = tendencies_nonlin(state_fft_np12_approx2)
        del(state_fft_np12_approx2)

        # # alternativelly, this
        # state_fft_temp += dt/3*exact2*tendencies_fft_3
        # state_fft_np1_approx = (exact*self.state_fft
        #                         + dt*exact2*tendencies_fft_3)
        # # or this (slightly faster...)

        datat = tendencies_fft_3

        state_fft_np1_approx = SetOfVariables(like=state_fft)
        datatemp2 = state_fft_np1_approx

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[ik, i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact[ik, i0, i1]*datas[ik, i0, i1] +
                        dt*exact2[ik, i0, i1]*datat[ik, i0, i1])

        del(tendencies_fft_3)
        tendencies_fft_4 = tendencies_nonlin(state_fft_np1_approx)
        del(state_fft_np1_approx)

        # # alternativelly, this
        # self.state_fft = state_fft_temp + dt/6*tendencies_fft_4
        # # or this (slightly faster... may be not...)

        datat = tendencies_fft_4

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datas[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/6*datat[ik, i0, i1])
