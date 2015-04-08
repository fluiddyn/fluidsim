

cimport numpy as np
import numpy as np
np.import_array()

from mpi4py import MPI
from mpi4py cimport MPI
from mpi4py.mpi_c cimport *

if MPI.COMM_WORLD.size == 1:
    raise ImportError('fftw2Dmpiccy only works if MPI.COMM_WORLD.size > 1.')

# fix a bug arising when using a recent version of mpi4py
cdef extern from 'mpi-compat.h': pass

from cpython cimport Py_INCREF

cimport libc
from libc.stddef cimport ptrdiff_t

from libc.stdlib cimport malloc, free

# this refers to the .pxd file
cimport libcfftw2dmpi

cdef extern from "numpy/arrayobject.h":
    object PyArray_SimpleNewFromData(int nd, int* dims,
                                     int typenum, void* data)

# we define python and c types for physical and Fourier spaces
DTYPEb = np.uint8
ctypedef np.uint8_t DTYPEb_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t
DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t
DTYPEc = np.complex128
ctypedef np.complex128_t DTYPEc_t

comm = MPI.COMM_WORLD

cdef class FFT2Dmpi(object):
    '''A FFT2Dmpi object is a wrapper to a c library
doing 2D parallele fft which uses MPI and sequencial functions
of the fftw library.
'''
    # number of nodes in the first and second dimensions
    cdef int N0, N1
    cdef public DTYPEb_t TRANSPOSED

    cdef libcfftw2dmpi.Util_fft uf

    # shape of the arrays in the physical and Fourier spaces,
    # for the sequential case:
    cdef public np.ndarray shapeX_seq, shapeK_seq, shapeK_gat
    # and for the parallel case:
    cdef public np.ndarray shapeX_loc, shapeK_loc

    # the communicator, nb of processus and rank of the processus
    cdef MPI.Comm comm
    cdef public int nb_proc, rank

    cdef public int iX0loc_start, iK1loc_start, iK0loc_start
    cdef public int iKxloc_start, iKyloc_start

    cdef public int idimx, idimy, idimkx, idimky

    cdef DTYPEc_t *carrayK
    cdef DTYPEf_t *carrayX

    def __init__(self, int N0, int N1):

        # info on MPI
        self.comm = comm
        self.nb_proc = self.comm.size
        self.rank = self.comm.Get_rank()

        if N0%2 != 0 or N1%2 != 0:
            raise ValueError('conditions n0, n1 have to be even')

        if N0%self.nb_proc != 0 or N1/2%self.nb_proc != 0:
            raise ValueError(
                'fftw2dmpiccy works only'
                ' if N0%self.nb_proc==0 and N1/2%self.nb_proc==0')

        self.N0 = N0
        self.N1 = N1

        # for sequenciel runs (not implemented with this library)
        # the data in K space is not transposed
        self.shapeX_seq = np.array([N0, N1])
        self.shapeK_seq = np.array([N0, N1/2+1])

        self.shapeK_gat = np.array([N1/2, N0])

        # the figures 0 and 1 correspond to the dimension in physical space,
        # the dimension 0 corresponds to the y-axes.
        # and the dimension 1 corresponds to the x-axes.
        self.idimx = 1
        self.idimy = 0
        self.idimkx = 0
        self.idimky = 1

        self.TRANSPOSED = 1

        # initialisation of the fft
        self.uf = libcfftw2dmpi.init_Util_fft(N0, N1)

        self.shapeX_loc = np.array([self.uf.nX0loc, self.uf.nX1])
        self.shapeK_loc = np.array([self.uf.nKxloc, self.uf.nKy])

        self.iX0loc_start = self.uf.nX0loc*self.rank
        self.iK0loc_start = self.uf.nK0loc*self.rank

        self.iK1loc_start = 0

        self.iKxloc_start = self.iK0loc_start
        self.iKyloc_start = self.iK1loc_start

        # allocation de carray
        self.carrayX = <DTYPEf_t *> malloc(self.shapeX_loc.prod()
                                           * sizeof(DTYPEf_t))
        self.carrayK = <DTYPEc_t *> malloc(self.shapeK_loc.prod()
                                           * sizeof(DTYPEc_t))


    # cpdef fft2d(self, np.ndarray[DTYPEf_t, ndim=2] ffX):
    cpdef fft2d(self, DTYPEf_t[:, :] ffX):
        cdef np.ndarray[DTYPEf_t, ndim=2, mode="c"] ffX_cont
        cdef np.ndarray[DTYPEc_t, ndim=2, mode="c"] ffK_cont
        ffX_cont = np.ascontiguousarray(ffX, dtype=DTYPEf)
        ffK_cont = np.empty(self.shapeK_loc, dtype=DTYPEc)
        libcfftw2dmpi.fft2D(self.uf, &ffX_cont[0,0], &ffK_cont[0,0])
        return ffK_cont

    # cpdef ifft2d(self, np.ndarray[DTYPEc_t, ndim=2] ffK):
    cpdef ifft2d(self, DTYPEc_t[:, :] ffK):
        cdef np.ndarray[DTYPEc_t, ndim=2, mode="c"] ffK_cont
        cdef np.ndarray[DTYPEf_t, ndim=2, mode="c"] ffX_cont
        ffK_cont = np.ascontiguousarray(ffK, dtype=DTYPEc)
        ffX_cont = np.empty(self.shapeX_loc, dtype=DTYPEf)
        libcfftw2dmpi.ifft2D(self.uf, &ffK_cont[0,0], &ffX_cont[0,0])
        return ffX_cont

    def __dealloc__(self):
        libcfftw2dmpi.destroy_Util_fft(self.uf)

    def describe(self):
        if self.rank == 0:
            print 'object of class Myfft2Dmpi'
            print 'N0 =', self.N0, 'N1 =', self.N1
            print 'nb_proc =', self.nb_proc,
            if self.nb_proc == 1:
                print '=> sequenciel version'
            else:
                print '=> parallel version (MPI)'

    def gather_Xspace(self, np.ndarray ff_loc,
                      root=None, type DTYPE=DTYPEf):
        cdef np.ndarray ff_seq

        # self.shapeX_loc is the same for all processes,
        # it is safe to use Allgather or Gather
        if root is None:
            ff_seq = np.empty(self.shapeX_seq, DTYPE)
            self.comm.Allgather(ff_loc, ff_seq)
        elif isinstance(root, int):
            ff_seq = None
            if self.rank == root:
                ff_seq = np.empty(self.shapeX_seq, DTYPE)
            self.comm.Gather(ff_loc, ff_seq, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_seq

    def scatter_Xspace(self, np.ndarray ff_seq,
                       int root=0, type DTYPE=DTYPEf):
        cdef np.ndarray ff_loc
        ff_loc = np.empty(self.shapeX_loc, dtype=DTYPE)
        # self.shapeX_loc is the same for all processes,
        # it is safe to use Scatter
        if isinstance(root, int):
            self.comm.Scatter(ff_seq, ff_loc, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_loc

    def gather_Kspace(self, np.ndarray ff_fft_loc, root=None, AS_SEQ=True):
        cdef np.ndarray ff_fft_seq
        cdef np.ndarray shapeK_temp
        rowtype = MPI.COMPLEX16.Create_contiguous(self.shapeK_loc[1])
        rowtype.Commit()
        if root is None:
            ff_fft_gat = np.empty(self.shapeK_gat, dtype=DTYPEc)
#            counts1 = self.comm.allgather(self.shapeK_loc[0])
#            print 'counts1 =', counts1
            counts = np.ones([self.nb_proc], dtype=int)*self.shapeK_loc[0]
#            print 'counts =', counts
            self.comm.Allgatherv(sendbuf=[ff_fft_loc, MPI.COMPLEX16],
                                 recvbuf=[ff_fft_gat, (counts, None), rowtype])
            if AS_SEQ:
                ff_fft_gat = ff_fft_gat.transpose()
                ff_fft_seq = np.empty(self.shapeK_seq, dtype=DTYPEc)
                ff_fft_seq[:, :-1] = ff_fft_gat
                ff_fft_seq[:, -1] = 0.
                result = ff_fft_seq
            else:
                result = ff_fft_gat
        elif isinstance(root, int):
            ff_fft_gat = None
            if self.rank == root:
                ff_fft_gat = np.empty(self.shapeK_gat, dtype=DTYPEc)
            counts = np.ones([self.nb_proc], dtype=int)*self.shapeK_loc[0]
            self.comm.Gatherv(sendbuf=[ff_fft_loc, MPI.COMPLEX16],
                              recvbuf=[ff_fft_gat, (counts, None), rowtype],
                              root=root)
            if AS_SEQ:
                if self.rank == root:
                    ff_fft_gat = ff_fft_gat.transpose()
                    ff_fft_seq = np.empty(self.shapeK_seq, dtype=DTYPEc)
                    ff_fft_seq[:, :-1] = ff_fft_gat
                    ff_fft_seq[:, -1] = 0.
                    result = ff_fft_seq
                else:
                    result = None
            else:
                result = ff_fft_gat
        else:
            raise ValueError('root should be an int')
        rowtype.Free()
        return result

    def scatter_Kspace(self, np.ndarray ff_fft_seq, int root=0,
                       AS_SEQ=True, type DTYPE=DTYPEc):
        cdef np.ndarray ff_fft_loc
        cdef np.ndarray shapeK_temp

        if not isinstance(root, int):
            raise ValueError('root should be an int')
        if AS_SEQ and root == self.rank:
            ff_fft_seq = ff_fft_seq[:, :-1]
            ff_fft_seq = ff_fft_seq.transpose()

        ff_fft_loc = np.empty(self.shapeK_loc, dtype=DTYPE)
        # self.shapeX_loc is the same for all processes,
        # it is safe to use Scatter
        self.comm.Scatter(ff_fft_seq, ff_fft_loc, root=root)
        return ff_fft_loc

    # functions for initialisation of field
    def constant_arrayK(self, value=None, dtype=complex, SHAPE='LOC'):
        """Return a constant array in spectral space."""
        if SHAPE == 'LOC':
            shapeK = self.shapeK_loc
        elif SHAPE == 'SEQ':
            shapeK = self.shapeK_seq
        else:
            raise ValueError('SHAPE should be ''LOC'' of ''SEQ''')
        if value is None:
            field_lm = np.empty(self.shapeK, dtype=dtype)
        elif value == 0:
            field_lm = np.zeros(self.shapeK, dtype=dtype)
        else:
            field_lm = value*np.ones(self.shapeK, dtype=dtype)
        return field_lm

    def constant_arrayX(self, value=None, dtype=DTYPEf, SHAPE='LOC'):
        """Return a constant array in real space."""
        if SHAPE == 'LOC':
            shapeX = self.shapeX_loc
        elif SHAPE == 'SEQ':
            shapeX = self.shapeX_seq
        else:
            raise ValueError('SHAPE should be ''LOC'' of ''SEQ''')
        if value is None:
            field = np.empty(shapeX, dtype=dtype)
        elif value == 0:
            field = np.zeros(shapeX, dtype=dtype)
        else:
            field = value*np.ones(shapeX, dtype=dtype)
        return field

    def project_fft_on_realX(self,
                             np.ndarray[DTYPEc_t, ndim=2] f_fft):
        """Project the given field in spectral space such as its
        inverse fft is a real field."""
        cdef np.uint32_t nky_seq
        cdef np.uint32_t iky_ky0, iky_kyM,  ikx_kx0, ikx_kxM,
        cdef np.uint32_t ikyp, ikyn
        cdef DTYPEc_t f_kp_kx0, f_kn_kx0, f_kp_kxM, f_knp_kxM

        nky_seq = self.shapeK_seq[0]
        iky_kyM = nky_seq/2

        if self.rank == 0:
            # first, some values have to be real
            f_fft[0, 0] = f_fft[0, 0].real
            f_fft[0, iky_kyM] = f_fft[0, iky_kyM].real

            # second, there are relations between some values
            for ikyp in xrange(1, iky_kyM):
                ikyn = nky_seq - ikyp

                f_kp_kx0 = f_fft[0, ikyp]
                f_kn_kx0 = f_fft[0, ikyn]

                f_fft[0, ikyp] = (f_kp_kx0+f_kn_kx0.conjugate()
                                  )/2
                f_fft[0, ikyn] = ((f_kp_kx0+f_kn_kx0.conjugate()
                                   )/2).conjugate()
