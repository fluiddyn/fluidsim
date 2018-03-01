"""
Time stepping Cython (:mod:`fluidsim.base.time_stepping.pseudo_spect_cy`)
=========================================================================

Provides:

.. autoclass:: ExactLinearCoefs
   :members:
   :private-members:

.. autoclass:: TimeSteppingPseudoSpectral
   :members:
   :private-members:

"""

from __future__ import absolute_import

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

from .pseudo_spect import ExactLinearCoefs as ExactLinearCoefsPurePython
from .pseudo_spect import TimeSteppingPseudoSpectral as \
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

    @cython.embedsignature(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _time_step_RK4_state_ndim3_freqlin_ndim2_float(self):
        """Advance in time *sim.state.state_spect* with the Runge-Kutta 4 method.

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
        state_spect = self.sim.state.state_spect

        nk = state_spect.shape[0]
        n0 = state_spect.shape[1]
        n1 = state_spect.shape[2]

        exact, exact2 = self.exact_linear_coefs.get_updated_coefs()

        tendencies_fft_1 = tendencies_nonlin()

        # # alternativelly, this
        # state_spect_temp = (self.state_spect + dt/6*tendencies_fft_1)*exact
        # state_spect_np12_approx1 = (
        #     self.state_spect + dt/2*tendencies_fft_1)*exact2
        # # or this (slightly faster...)

        datas = state_spect
        datat = tendencies_fft_1

        state_spect_temp = SetOfVariables(like=state_spect)
        datatemp = state_spect_temp

        state_spect_np12_approx1 = SetOfVariables(like=state_spect)
        datatemp2 = state_spect_np12_approx1

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/6*datat[ik, i0, i1])*exact[i0, i1]
                    datatemp2[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])*exact2[i0, i1]

        tendencies_fft_2 = tendencies_nonlin(
            state_spect_np12_approx1, old=tendencies_fft_1)

        # # alternativelly, this
        # state_spect_temp += dt/3*exact2*tendencies_fft_2
        # state_spect_np12_approx2 = (exact2*self.state_spect
        #                           + dt/2*tendencies_fft_2)
        # # or this (slightly faster...)

        datat = tendencies_fft_2

        state_spect_np12_approx2 = state_spect_np12_approx1
        del(state_spect_np12_approx1)
        
        datatemp2 = state_spect_np12_approx2

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact2[i0, i1]*datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])

        tendencies_fft_3 = tendencies_nonlin(
            state_spect_np12_approx2, old=tendencies_fft_2)

        # # alternativelly, this
        # state_spect_temp += dt/3*exact2*tendencies_fft_3
        # state_spect_np1_approx = (exact*self.state_spect
        #                         + dt*exact2*tendencies_fft_3)
        # # or this (slightly faster...)

        datat = tendencies_fft_3

        state_spect_np1_approx = state_spect_np12_approx2
        del(state_spect_np12_approx2)
        datatemp2 = state_spect_np1_approx

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact[i0, i1]*datas[ik, i0, i1] +
                        dt*exact2[i0, i1]*datat[ik, i0, i1])

        tendencies_fft_4 = tendencies_nonlin(
            state_spect_np1_approx, old=tendencies_fft_3)
        del(state_spect_np1_approx)

        # # alternativelly, this
        # self.state_spect = state_spect_temp + dt/6*tendencies_fft_4
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
        """Advance in time *sim.state.state_spect* with the Runge-Kutta 4 method.

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

        state_spect = self.sim.state.state_spect
        datas = state_spect
        nk = datas.shape[0]
        n0 = datas.shape[1]
        n1 = datas.shape[2]

        exact, exact2 = self.exact_linear_coefs.get_updated_coefs()

        tendencies_fft_1 = tendencies_nonlin()

        # # alternativelly, this
        # state_spect_temp = (self.state_spect + dt/6*tendencies_fft_1)*exact
        # state_spect_np12_approx1 = (self.state_spect
        #                           + dt/2*tendencies_fft_1)*exact2
        # # or this (slightly faster...)

        datat = tendencies_fft_1

        state_spect_temp = SetOfVariables(like=state_spect)
        datatemp = state_spect_temp

        state_spect_np12_approx1 = SetOfVariables(like=state_spect)
        datatemp2 = state_spect_np12_approx1

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/6*datat[ik, i0, i1])*exact[ik, i0, i1]
                    datatemp2[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])*exact2[ik, i0, i1]

        tendencies_fft_2 = tendencies_nonlin(
            state_spect_np12_approx1, old=tendencies_fft_1)

        # # alternativelly, this
        # state_spect_temp += dt/3*exact2*tendencies_fft_2
        # state_spect_np12_approx2 = (exact2*self.state_spect
        #                           + dt/2*tendencies_fft_2)
        # # or this (slightly faster...)

        datat = tendencies_fft_2

        state_spect_np12_approx2 = state_spect_np12_approx1
        del(state_spect_np12_approx1)
        datatemp2 = state_spect_np12_approx2

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[ik, i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact2[ik, i0, i1]*datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])

        tendencies_fft_3 = tendencies_nonlin(
            state_spect_np12_approx2, old=tendencies_fft_2)

        # # alternativelly, this
        # state_spect_temp += dt/3*exact2*tendencies_fft_3
        # state_spect_np1_approx = (exact*self.state_spect
        #                         + dt*exact2*tendencies_fft_3)
        # # or this (slightly faster...)

        datat = tendencies_fft_3

        state_spect_np1_approx = state_spect_np12_approx2
        del(state_spect_np12_approx2)
        datatemp2 = state_spect_np1_approx

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[ik, i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact[ik, i0, i1]*datas[ik, i0, i1] +
                        dt*exact2[ik, i0, i1]*datat[ik, i0, i1])

        tendencies_fft_4 = tendencies_nonlin(
            state_spect_np1_approx, old=tendencies_fft_3)
        del(state_spect_np1_approx)

        # # alternativelly, this
        # self.state_spect = state_spect_temp + dt/6*tendencies_fft_4
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
        """Advance in time *sim.state.state_spect* with the Runge-Kutta 4 method.

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

        state_spect = self.sim.state.state_spect
        datas = state_spect
        nk = datas.shape[0]
        n0 = datas.shape[1]
        n1 = datas.shape[2]

        exact, exact2 = self.exact_linear_coefs.get_updated_coefs()

        tendencies_fft_1 = tendencies_nonlin()

        # # alternativelly, this
        # state_spect_temp = (self.state_spect + dt/6*tendencies_fft_1)*exact
        # state_spect_np12_approx1 = (self.state_spect
        #                           + dt/2*tendencies_fft_1)*exact2
        # # or this (slightly faster...)

        datat = tendencies_fft_1

        state_spect_temp = SetOfVariables(like=state_spect)
        datatemp = state_spect_temp

        state_spect_np12_approx1 = SetOfVariables(like=state_spect)
        datatemp2 = state_spect_np12_approx1

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/6*datat[ik, i0, i1])*exact[ik, i0, i1]
                    datatemp2[ik, i0, i1] = (
                        datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])*exact2[ik, i0, i1]

        tendencies_fft_2 = tendencies_nonlin(
            state_spect_np12_approx1, old=tendencies_fft_1)

        # # alternativelly, this
        # state_spect_temp += dt/3*exact2*tendencies_fft_2
        # state_spect_np12_approx2 = (exact2*self.state_spect
        #                           + dt/2*tendencies_fft_2)
        # # or this (slightly faster...)

        datat = tendencies_fft_2

        state_spect_np12_approx2 = state_spect_np12_approx1
        del(state_spect_np12_approx1)
        datatemp2 = state_spect_np12_approx2

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[ik, i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact2[ik, i0, i1]*datas[ik, i0, i1] +
                        dt/2*datat[ik, i0, i1])

        tendencies_fft_3 = tendencies_nonlin(
            state_spect_np12_approx2, old=tendencies_fft_2)

        # # alternativelly, this
        # state_spect_temp += dt/3*exact2*tendencies_fft_3
        # state_spect_np1_approx = (exact*self.state_spect
        #                         + dt*exact2*tendencies_fft_3)
        # # or this (slightly faster...)

        datat = tendencies_fft_3

        state_spect_np1_approx = state_spect_np12_approx2
        del(state_spect_np12_approx2)
        datatemp2 = state_spect_np1_approx

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    datatemp[ik, i0, i1] = (
                        datatemp[ik, i0, i1] +
                        dt/3*exact2[ik, i0, i1]*datat[ik, i0, i1])
                    datatemp2[ik, i0, i1] = (
                        exact[ik, i0, i1]*datas[ik, i0, i1] +
                        dt*exact2[ik, i0, i1]*datat[ik, i0, i1])

        tendencies_fft_4 = tendencies_nonlin(
            state_spect_np1_approx, old=tendencies_fft_3)
        del(state_spect_np1_approx)

        # # alternativelly, this
        # self.state_spect = state_spect_temp + dt/6*tendencies_fft_4
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
    def _time_step_RK4_state_ndim4_freqlin_ndim3_float(self):
        """Advance in time *sim.state.state_spect* with the Runge-Kutta 4 method.

        See :ref:`the pure python RK4 function <rk4timescheme>` for the
        presentation of the time scheme.

        For this function, the coefficient :math:`\sigma` is real and
        represents the dissipation.

        """
        # cdef DTYPEf_t dt = self.deltat
        cdef double dt = self.deltat

        cdef Py_ssize_t i0, i1, i2, ik, nk, n0, n1, n2

        # cdef np.ndarray[DTYPEf_t, ndim=2] exact, exact2
        # This is strange, if I use DTYPEf_t and complex.h => bug
        cdef np.ndarray[double, ndim=3] exact, exact2

        cdef np.ndarray[DTYPEc_t, ndim=4] datas, datat
        cdef np.ndarray[DTYPEc_t, ndim=4] datatemp, datatemp2

        tendencies_nonlin = self.sim.tendencies_nonlin
        state_spect = self.sim.state.state_spect

        nk = state_spect.shape[0]
        n0 = state_spect.shape[1]
        n1 = state_spect.shape[2]
        n2 = state_spect.shape[3]

        exact, exact2 = self.exact_linear_coefs.get_updated_coefs()

        tendencies_fft_1 = tendencies_nonlin()

        # # alternativelly, this
        # state_spect_temp = (self.state_spect + dt/6*tendencies_fft_1)*exact
        # state_spect_np12_approx1 = (
        #     self.state_spect + dt/2*tendencies_fft_1)*exact2
        # # or this (slightly faster...)

        datas = state_spect
        datat = tendencies_fft_1

        state_spect_temp = SetOfVariables(like=state_spect)
        datatemp = state_spect_temp

        state_spect_np12_approx1 = SetOfVariables(like=state_spect)
        datatemp2 = state_spect_np12_approx1

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    for i2 in xrange(n2):
                        datatemp[ik, i0, i1, i2] = (
                            datas[ik, i0, i1, i2] +
                            dt/6*datat[ik, i0, i1, i2])*exact[i0, i1, i2]
                        datatemp2[ik, i0, i1, i2] = (
                            datas[ik, i0, i1, i2] +
                            dt/2*datat[ik, i0, i1, i2])*exact2[i0, i1, i2]

        tendencies_fft_2 = tendencies_nonlin(
            state_spect_np12_approx1, old=tendencies_fft_1)

        # # alternativelly, this
        # state_spect_temp += dt/3*exact2*tendencies_fft_2
        # state_spect_np12_approx2 = (exact2*self.state_spect
        #                           + dt/2*tendencies_fft_2)
        # # or this (slightly faster...)

        datat = tendencies_fft_2

        state_spect_np12_approx2 = state_spect_np12_approx1
        del(state_spect_np12_approx1)
        datatemp2 = state_spect_np12_approx2

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    for i2 in xrange(n2):
                        datatemp[ik, i0, i1, i2] = (
                            datatemp[ik, i0, i1, i2] +
                            dt/3*exact2[i0, i1, i2]*datat[ik, i0, i1, i2])
                        datatemp2[ik, i0, i1, i2] = (
                            exact2[i0, i1, i2]*datas[ik, i0, i1, i2] +
                            dt/2*datat[ik, i0, i1, i2])

        tendencies_fft_3 = tendencies_nonlin(
            state_spect_np12_approx2, old=tendencies_fft_2)

        # # alternativelly, this
        # state_spect_temp += dt/3*exact2*tendencies_fft_3
        # state_spect_np1_approx = (exact*self.state_spect
        #                         + dt*exact2*tendencies_fft_3)
        # # or this (slightly faster...)

        datat = tendencies_fft_3

        state_spect_np1_approx = state_spect_np12_approx2
        del(state_spect_np12_approx2)
        datatemp2 = state_spect_np1_approx

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    for i2 in xrange(n2):
                        datatemp[ik, i0, i1, i2] = (
                            datatemp[ik, i0, i1, i2] +
                            dt/3*exact2[i0, i1, i2]*datat[ik, i0, i1, i2])
                        datatemp2[ik, i0, i1, i2] = (
                            exact[i0, i1, i2]*datas[ik, i0, i1, i2] +
                            dt*exact2[i0, i1, i2]*datat[ik, i0, i1, i2])

        tendencies_fft_4 = tendencies_nonlin(
            state_spect_np1_approx, old=tendencies_fft_3)
        del(state_spect_np1_approx)

        # # alternativelly, this
        # self.state_spect = state_spect_temp + dt/6*tendencies_fft_4
        # # or this (slightly faster... may be not...)

        datat = tendencies_fft_4

        for ik in xrange(nk):
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    for i2 in xrange(n2):
                        datas[ik, i0, i1, i2] = (
                            datatemp[ik, i0, i1, i2] +
                            dt/6*datat[ik, i0, i1, i2])
