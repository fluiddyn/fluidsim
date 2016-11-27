"""Miscellaneous functions (:mod:`fluidsim.operators.miscellaneous)
===================================================================

.. currentmodule:: fluidsim.operators.miscellaneous

This module is written in Cython and provides:



"""
# DEF MPI4PY = 0

import cython

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
    # from mpi4py.mpi_c cimport *
    # for mpi4py > 2.0
    from mpi4py.libmpi cimport *
    
    # solve an incompatibility between openmpi and mpi4py versions
    cdef extern from 'mpi-compat.h': pass


cdef extern from "complex.h": pass
    # double cabs(double complex) nogil

# we define python and c types for physical and Fourier spaces
DTYPEb = np.uint8
ctypedef np.uint8_t DTYPEb_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t
DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t
DTYPEc = np.complex128
ctypedef np.complex128_t DTYPEc_t


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_correl4(np.ndarray[DTYPEc_t, ndim=2] q_fftt,
                    np.ndarray[DTYPEi_t, ndim=1] iomegas1,
                    int nb_omegas, int nb_xs_seq):
    r"""Compute the correlations 4.

    .. math::
       C_4(\omega_1, \omega_2, \omega_3, \omega_4) =
       \langle
       \tilde w(\omega_1, \mathbf{x})
       \tilde w(\omega_2, \mathbf{x})
       \tilde w(\omega_3, \mathbf{x})^*
       \tilde w(\omega_4, \mathbf{x})^*
       \rangle_\mathbf{x},

    where

    .. math::
       \omega_2 = \omega_3 + \omega_4 - \omega_1

    and :math:`\omega_1 > 0`, :math:`\omega_3 > 0` and
    :math:`\omega_4 > 0`. Thus, this function produces an array
    :math:`C_4(\omega_1, \omega_3, \omega_4)`.

    """
    cdef DTYPEi_t i1, io1, io2, io3, io4, ix
    cdef np.ndarray[DTYPEc_t, ndim=2] q_fftt_conj = np.conj(q_fftt)
    cdef np.ndarray[DTYPEc_t, ndim=3] corr4

    cdef int nx = q_fftt.shape[0]
    cdef int n0 = iomegas1.shape[0]

    corr4 = np.zeros([len(iomegas1), nb_omegas, nb_omegas], dtype=DTYPEc)

    for i1 in range(n0):
        io1 = iomegas1[i1]
        # this loop could be parallelized (OMP)
        for io3 in range(nb_omegas):
            # we use the symmetry omega_3 <--> omega_4
            for io4 in range(io3+1):
                io2 = io3 + io4 - io1
                if io2 < 0:
                    io2 = -io2
                    for ix in range(nx):
                        corr4[i1, io3, io4] += (
                            q_fftt[ix, io1] *
                            q_fftt_conj[ix, io3] *
                            q_fftt_conj[ix, io4] *
                            q_fftt_conj[ix, io2])
                elif io2 >= nb_omegas:
                    io2 = 2*nb_omegas-1-io2
                    for ix in range(nx):
                        corr4[i1, io3, io4] += (
                            q_fftt[ix, io1] *
                            q_fftt_conj[ix, io3] *
                            q_fftt_conj[ix, io4] *
                            q_fftt_conj[ix, io2])
                else:
                    for ix in range(nx):
                        corr4[i1, io3, io4] += (
                            q_fftt[ix, io1] *
                            q_fftt_conj[ix, io3] *
                            q_fftt_conj[ix, io4] *
                            q_fftt[ix, io2])
                # symmetry omega_3 <--> omega_4:
                corr4[i1, io4, io3] = corr4[i1, io3, io4]

    if nb_proc > 1:
        # reduce SUM for mean:
        corr4 = comm.reduce(corr4, op=MPI.SUM, root=0)

    if rank == 0:
        corr4 /= nb_xs_seq
        return corr4


def compute_correl2(np.ndarray[DTYPEc_t, ndim=2] q_fftt,
                    np.ndarray[DTYPEi_t, ndim=1] iomegas1,
                    int nb_omegas, int nb_xs_seq):
    r"""Compute the correlations 2.

    .. math::
       C_2(\omega_1, \omega_2) =
       \langle
       \tilde w(\omega_1, \mathbf{x})
       \tilde w(\omega_2, \mathbf{x})^*
       \rangle_\mathbf{x},

    where :math:`\omega_1 = \omega_2`. Thus, this function
    produces an array :math:`C_2(\omega)`.

    """
    cdef DTYPEi_t io3, io4, ix
    cdef np.ndarray[DTYPEc_t, ndim=2] q_fftt_conj = np.conj(q_fftt)
    cdef np.ndarray[DTYPEc_t, ndim=2] corr2
    cdef int nx = q_fftt.shape[0]

    corr2 = np.zeros([nb_omegas, nb_omegas], dtype=DTYPEc)

    for io3 in range(nb_omegas):
        for io4 in range(io3+1):
            for ix in range(nx):
                corr2[io3, io4] += (
                    q_fftt[ix, io3] * q_fftt_conj[ix, io4])
            corr2[io4, io3] = np.conj(corr2[io3, io4])

    if nb_proc > 1:
        # reduce SUM for mean:
        corr2 = comm.reduce(corr2, op=MPI.SUM, root=0)

    if rank == 0:
        corr2 /= nb_xs_seq
        return corr2
