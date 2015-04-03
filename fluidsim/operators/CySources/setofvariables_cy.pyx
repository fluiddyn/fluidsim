"""Variable container (:mod:`fluidsim.base.setofvariables`)
======================================================================

This module is written in cython and provides:

.. currentmodule:: fluidsim.base.setofvariables

Provides:

.. autoclass:: SetOfVariables
   :members:
   :private-members:

"""

# DEF MPI4PY = 0

cimport numpy as np
import numpy as np
np.import_array()

try:
    from mpi4py import MPI
except ImportError:
    nb_proc = 1
    rank = 0
else:
    comm = MPI.COMM_WORLD
    nb_proc = comm.size
    rank = comm.Get_rank()

IF MPI4PY:
    from mpi4py cimport MPI
    from mpi4py.mpi_c cimport *

    # solve an incompatibility between openmpi and mpi4py versions
    cdef extern from 'mpi-compat.h': pass


from time import time, sleep
import datetime
import os
import matplotlib.pyplot as plt
import cython

from libc.math cimport exp


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


class SetOfVariables(object):
    """Gather a set of variables in a Numpy array.


    """
    __array_priority__ = 100

    @property
    def nbytes(self):
        return self.data.nbytes

    def __init__(self,
                 keys=None, shape1var=None,
                 dtype=None, info=None,
                 like=None, value=None):
        if like is not None:
            keys = like.keys
            self.nb_variables = like.data.shape[0]
            shape1var = like.data.shape[1:]
            if dtype is None:
                dtype = like.data.dtype
            if info is None:
                info = like.info
        else:
            if dtype is None:
                dtype = np.float64
            keys.sort()
            self.nb_variables = len(keys)

        self.info = info
        self.keys = keys
        shape = [self.nb_variables]
        shape.extend(shape1var)
        if value is None:
            self.data = np.empty(shape, dtype=dtype)
        elif value == 0:
            self.data = np.zeros(shape, dtype=dtype)
        else:
            self.data = value*np.ones(shape, dtype=dtype)

        dimension_space = len(shape1var)
        if dimension_space == 1:
            self.dealiasing = self._dealiasing1d
        elif dimension_space == 2:
            self.dealiasing = self._dealiasing2d
        elif dimension_space == 3:
            self.dealiasing = self._dealiasing3d
        else:
            raise ValueError(
                'Space dimension {} not implemented in SetOfVariables.'.format(
                    dimension_space))

    def __getitem__(self, key):
        ik = self.keys.index(key)
        return self.data[ik]

    def __setitem__(self, key, value):
        ik = self.keys.index(key)
        self.data[ik][:] = value

    def __add__(self, other):
        if isinstance(other, self.__class__):
            dtype_new = max_dtype(self.data, other.data)
            obj_result = self.__class__(like=other, dtype=dtype_new)
            obj_result.data = self.data + other.data
        elif isinstance(other, (int, float, complex)):
            dtype_new = max_dtype(self.data, other)
            obj_result = self.__class__(like=self, dtype=dtype_new)
            obj_result.data = self.data + other
        return obj_result
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            dtype_new = max_dtype(self.data, other.data)
            if dtype_new != self.data.dtype:
                obj_result = self.__class__(
                    like=self, dtype=dtype_new)
            else:
                obj_result = self
            obj_result.data += other.data
        elif isinstance(other, (int, float, complex)):
            dtype_new = max_dtype(self.data, other)
            if dtype_new != self.data.dtype:
                obj_result = self.__class__(
                    like=self, dtype=dtype_new)
            else:
                obj_result = self
            obj_result.data += other.data
        return obj_result

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            dtype_new = max_dtype(self.data, other.data)
            obj_result = self.__class__(like=other, dtype=dtype_new)
            obj_result.data = self.data - other.data
        elif isinstance(other, (int, float, complex)):
            dtype_new = max_dtype(self.data, other)
            obj_result = self.__class__(like=self, dtype=dtype_new)
            obj_result.data = self.data - other
        return obj_result

    def __mul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            dtype_new = max_dtype(self.data, other)
            obj_result = self.__class__(like=self, dtype=dtype_new)
            obj_result.data = other*self.data
        return obj_result
    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            dtype_new = max_dtype(self.data, other)
            obj_result = self.__class__(like=self, dtype=dtype_new)
            obj_result.data = self.data/other
        return obj_result

    def initialize(self, value=0.):
        self.data = value*np.ones(self.data.shape,
                                  dtype=self.data.dtype)

    def _dealiasing1d(self, DTYPEb_t[:] where_dealiased):
        cdef DTYPEc_t[:, :] data = self.data
        cdef Py_ssize_t ik, nk, i0, n0

        nk = self.nb_variables
        n0 = data.shape[1]

        for i0 in xrange(n0):
            if where_dealiased[i0]:
                for ik in xrange(nk):
                    data[ik, i0] = 0.

    def _dealiasing2d(self, DTYPEb_t[:, :] where_dealiased):
        cdef DTYPEc_t[:, :, :] data = self.data
        cdef Py_ssize_t ik, nk, i0, n0, i1, n1

        nk = self.nb_variables
        n0 = data.shape[1]
        n1 = data.shape[2]

        for i0 in xrange(n0):
            for i1 in xrange(n1):
                if where_dealiased[i0, i1]:
                    for ik in xrange(nk):
                        data[ik, i0, i1] = 0.

    def _dealiasing3d(self, DTYPEb_t[:, :, :] where_dealiased):
        cdef DTYPEc_t[:, :, :, :] data = self.data
        cdef Py_ssize_t ik, nk, i0, n0, i1, n1, i2, n2

        nk = self.nb_variables
        n0 = data.shape[1]
        n1 = data.shape[2]
        n2 = data.shape[3]

        for i0 in xrange(n0):
            for i1 in xrange(n1):
                for i2 in xrange(n2):
                    if where_dealiased[i0, i1, i2]:
                        for ik in xrange(nk):
                            data[ik, i0, i1, i2] = 0.


def max_dtype(A, B):
    '''Return the dtype of the result of an operation involving A and B.'''
    # it would be better to just use
    try:
        # this function is only available in numpy 1.6
        return np.result_type(A, B)
    except AttributeError:
        if isinstance(A, np.ndarray):
            dtypeA = A.dtype
        else:
            dtypeA = np.array(A).dtype

        if isinstance(B, np.ndarray):
            dtypeB = B.dtype
        else:
            dtypeB = np.array(B).dtype

        if dtypeA <= dtypeB:
            return dtypeB
        else:
            return dtypeA
