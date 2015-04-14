

from __future__ import division, print_function

from time import sleep

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
    cdef extern from 'mpi-compat.h':
        pass

from cpython cimport Py_INCREF

cimport libc
from libc.stddef cimport ptrdiff_t

cimport fftw3
from fftw3 cimport fftw_iodim, FFTW_FORWARD, FFTW_BACKWARD, \
    FFTW_MEASURE, FFTW_DESTROY_INPUT, FFTW_UNALIGNED, \
    FFTW_CONSERVE_MEMORY, FFTW_EXHAUSTIVE, FFTW_PRESERVE_INPUT, \
    FFTW_PATIENT, FFTW_ESTIMATE, FFTW_MPI_TRANSPOSED_IN, \
    FFTW_MPI_TRANSPOSED_OUT, fftw_plan, FFTW_MPI_DEFAULT_BLOCK

IF MPI4PY:
    cimport fftw3mpi


fftw_flags = {'FFTW_CONSERVE_MEMORY': FFTW_CONSERVE_MEMORY,
              'FFTW_DESTROY_INPUT': FFTW_DESTROY_INPUT,
              'FFTW_ESTIMATE': FFTW_ESTIMATE,
              'FFTW_EXHAUSTIVE': FFTW_EXHAUSTIVE,
              'FFTW_MEASURE': FFTW_MEASURE,
              'FFTW_MPI_TRANSPOSED_IN': FFTW_MPI_TRANSPOSED_IN,
              'FFTW_MPI_TRANSPOSED_OUT': FFTW_MPI_TRANSPOSED_OUT,
              'FFTW_PATIENT': FFTW_PATIENT,
              'FFTW_PRESERVE_INPUT': FFTW_PRESERVE_INPUT,
              'FFTW_UNALIGNED': FFTW_UNALIGNED}

from cpython.ref cimport PyTypeObject

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject * subtype, np.dtype descr,
                                int nd, np.npy_intp* dims, np.npy_intp* strides,
                                void* data, int flags, object obj)
    object PyArray_SimpleNewFromData(int nd, int* dims, int typenum,void* data)


# we define python and c types for physical and Fourier spaces
DTYPEb = np.uint8
ctypedef np.uint8_t DTYPEb_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t
DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t
DTYPEc = np.complex128
ctypedef np.complex128_t DTYPEc_t


cdef class FFT2Dmpi(object):
    """The FFT2Dmpi class is a cython wrapper for the 2D fast Fourier
    transform (sequencial and MPI) of the fftw library."""
    # number of nodes in the first and second dimensions
    cdef int n0, n1
    # flags for fftw
    cdef int flags
    cdef public DTYPEb_t TRANSPOSED, SEQUENCIAL

    # coef for normalization
    cdef int coef_norm

    cdef np.ndarray arrayX, arrayK

    cdef complex *carrayK
    cdef DTYPEf_t *carrayX

    cdef fftw_plan plan_forward
    cdef fftw_plan plan_backward

    # shape of the arrays in the physical and Fourier spaces,
    # for the sequential case:
    cdef public np.ndarray shapeX_seq, shapeK_seq, shapeK_gat
    # and for the parallel case:
    cdef public np.ndarray shapeX_loc, shapeK_loc, shapeX_locpad

    # the communicator, nb of processus and rank of the processus
    IF MPI4PY:
        cdef MPI.Comm comm

    cdef public int nb_proc, rank

    cdef public ptrdiff_t iX0loc_start, iKxloc_start, iKyloc_start

    cdef public int idimx, idimy, idimkx, idimky

    cdef size_t n_alloc_local

    def __init__(self, int n0, int n1, flags=['FFTW_MEASURE'],
                 TRANSPOSED=True, SEQUENCIAL=None):

        if TRANSPOSED is None:
            TRANSPOSED = True

        if nb_proc == 1 or SEQUENCIAL:
            self.SEQUENCIAL = True
            if SEQUENCIAL and rank == 0 and nb_proc > 1:
                print('    sequencial version even though self.nb_proc > 1')
        else:
            self.SEQUENCIAL = False

        # info on MPI
        self.nb_proc = nb_proc
        self.rank = rank
        if self.nb_proc > 1:
            self.comm = comm

        if n0 % 2 != 0 or n1 % 2 != 0:
            raise ValueError('conditions n0 and n1 even not fulfill')

        if not self.SEQUENCIAL and n0//2 + 1 < nb_proc:
            raise ValueError('condition nx//2+1 >= nb_proc not fulfill')

        self.n0 = n0
        self.n1 = n1

        self.shapeX_seq = np.array([n0, n1])
        self.shapeK_seq = np.array([n0, n1//2+1])

        # print('shapeX_seq:', shapeX_seq, '\nshapeK_seq:', shapeK_seq)

        if self.nb_proc == 1 or SEQUENCIAL:
            TRANSPOSED = False

        self.TRANSPOSED = TRANSPOSED

        # we consider that the first dimension corresponds to the x-axes.
        # and the second dimension corresponds to the y-axes.
        self.idimx = 1
        self.idimy = 0

        if self.TRANSPOSED:
            self.idimkx = 0
            self.idimky = 1
            self.shapeK_gat = np.array([n1//2+1, n0])
        else:
            self.idimkx = 1
            self.idimky = 0
            self.shapeK_gat = np.array([n0, n1//2+1])

        for f in flags:
            self.flags = self.flags | fftw_flags[f]

        self.coef_norm = n0*n1

        # Allocate the carrays and create the plans
        # and create the np arrays pointing to the carrays
        if self.nb_proc == 1 or SEQUENCIAL:
            self.init_seq()
        else:
            self.init_parall()

    cdef init_seq(self):
        """
        Allocate the carrays, create the plans (for sequential FFTW)
        and create np arrays pointing on the carrays
        """

        # print('init_seq:', self.shapeX_seq, self.shapeK_seq)

        # from fluiddyn.util.debug_with_ipython import ipydebug
        # ipydebug()

        self.shapeX_loc = self.shapeX_seq
        self.shapeK_loc = self.shapeK_seq
        self.iKxloc_start = 0
        self.iKyloc_start = 0
        self.n_alloc_local = self.shapeK_loc.prod()

        # print('self.n_alloc_local:', self.n_alloc_local)

        self.carrayK = fftw3.fftw_alloc_complex(<size_t> self.shapeK_loc.prod())
        self.carrayX = fftw3.fftw_alloc_real(<size_t>  self.shapeX_loc.prod())

        # print('after alloc')

        self.plan_forward = fftw3.fftw_plan_dft_r2c_2d(self.n0, self.n1,
                                                       <double*> self.carrayX,
                                                       <complex*> self.carrayK,
                                                       self.flags)
        self.plan_backward = fftw3.fftw_plan_dft_c2r_2d(self.n0, self.n1,
                                                        <complex*> self.carrayK,
                                                        <double*> self.carrayX,
                                                        self.flags)

        # print(self.n0, self.n1, self.shapeX_loc.data)
        # print('after planning, self.flags:', self.flags)

        self.arrayX = PyArray_SimpleNewFromData(
            <int> 2, <np.npy_intp *> self.shapeX_loc.data,
            np.NPY_FLOAT64, <void*> self.carrayX)
        # print('after first PyArray_SimpleNewFromData')

        self.arrayK = PyArray_SimpleNewFromData(
            <int> 2, <np.npy_intp *> self.shapeK_loc.data,
            np.NPY_COMPLEX128, <void*> self.carrayK)

        # print('after PyArray_SimpleNewFromData')

    IF MPI4PY:

        cdef init_parall(self):
            """
            Allocate the carrays, create the plans (for MPI FFTW)
            and create the np arrays pointing to the carrays
            """
            cdef MPI_Comm c_comm = self.comm.ob_mpi
            cdef int flags_temp
            cdef size_t n_alloc_local
            cdef ptrdiff_t nX0loc, nKxloc

    #        if self.rank==0: print 'self.init_parall()'
            fftw3mpi.fftw_mpi_init()

            if not self.TRANSPOSED:
                n_alloc_local = fftw3mpi.fftw_mpi_local_size_2d(
                    <size_t> self.n0, <size_t> self.n1//2+1,
                    c_comm,
                    &nX0loc, &self.iX0loc_start)
                self.shapeK_loc = np.array([nX0loc, self.n1//2+1])
                self.iKxloc_start = 0
                self.iKyloc_start = self.iX0loc_start
            else:
                n_alloc_local = fftw3mpi.fftw_mpi_local_size_2d_transposed(
                    <size_t> self.n0, <size_t> self.n1//2+1,
                    c_comm,
                    &nX0loc, &self.iX0loc_start,
                    &nKxloc, &self.iKxloc_start)
                self.shapeK_loc = np.array([nKxloc, self.n0])
                self.iKyloc_start = 0

            self.n_alloc_local = n_alloc_local

            self.shapeX_loc = np.array([nX0loc, self.n1])

            self.shapeX_locpad     = self.shapeX_loc.copy()
            self.shapeX_locpad[-1] = 2*(self.shapeX_loc[-1]//2+1)

            self.carrayK = fftw3.fftw_alloc_complex(n_alloc_local)
            self.carrayX = fftw3.fftw_alloc_real(2 * n_alloc_local)

    ####        for r in xrange(self.nb_proc):
    ####            self.comm.barrier()
    ####            sleep(0.05)
    ####            if self.rank==r:
    ####                print  'rank =', self.rank, 'n_alloc_local =', n_alloc_local,\
    ####                        'other:', nX0loc, self.iX0loc_start,\
    ####                         self.iKxloc_start, self.iKyloc_start
    ####                print  'self.shapeX_locpad =', self.shapeX_locpad
    ####                print  'self.shapeX_loc    =', self.shapeX_loc
    ####                print  'self.shapeK_loc    =', self.shapeK_loc

            if self.TRANSPOSED:
                flags_temp = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_OUT']
            else:
                flags_temp = self.flags
            self.plan_forward = fftw3mpi.fftw_mpi_plan_dft_r2c_2d(  
                <size_t> self.n0, <size_t> self.n1,
                <double*> self.carrayX,
                <complex*> self.carrayK,
                c_comm,
                flags_temp)

            if self.TRANSPOSED:
                flags_temp = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_IN']
            else:
                flags_temp = self.flags
            self.plan_backward = fftw3mpi.fftw_mpi_plan_dft_c2r_2d(
                <size_t> self.n0, <size_t> self.n1,
                <complex*> self.carrayK,
                <double*> self.carrayX,
                c_comm,
                flags_temp)

            # create the python arrays (arrayX and arrayK) pointing
            # toward carrayX and carrayK
            cdef np.dtype npDTYPEf = np.dtype('float64')
            Py_INCREF(npDTYPEf)
            cdef np.ndarray[DTYPEi_t, ndim=1] stridesX
            # stridesX en nb elements
            stridesX = np.array([self.shapeX_locpad[1], 1])
            # stridesX en octet
            stridesX *= npDTYPEf.itemsize
            cdef int ndim = 2
            self.arrayX = PyArray_NewFromDescr(
                <PyTypeObject *> np.ndarray,
                npDTYPEf, ndim,
                <np.npy_intp *> self.shapeX_loc.data,
                <np.npy_intp *> stridesX.data,
                <void *> self.carrayX,
                np.NPY_DEFAULT, None)

            self.arrayK = PyArray_SimpleNewFromData(
                <int> 2, <np.npy_intp *> self.shapeK_loc.data,
                np.NPY_COMPLEX128, <void*> self.carrayK)

    # cpdef fft2d(self, np.ndarray[DTYPEf_t, ndim=2] ff):
    cpdef fft2d(self, DTYPEf_t[:, :] ff):
        self.arrayX[:] = ff
        fftw3.fftw_execute(self.plan_forward)
        return self.arrayK/self.coef_norm

    # cpdef ifft2d(self, np.ndarray[DTYPEc_t, ndim=2] ff_fft):
    cpdef ifft2d(self, DTYPEc_t[:, :] ff_fft):
        """Inverse Fast Fourier Transform 2D

        This is THE function where most of the time is spent !
        """
        # print ff_fft
        self.arrayK[:] = ff_fft
        # self.print_carrayK()
        fftw3.fftw_execute(self.plan_backward)
        # self.print_carrayX()
        # print self.arrayX
#### result = self.arrayX.copy()     # BUG with pelvoux !!! (python 2.6)
#### result = self.arrayX*1          # works, but maybe slower ?
        return self.arrayX.copy()

    def __dealloc__(self):
        if self.nb_proc == 1:
            fftw3.fftw_destroy_plan(self.plan_forward)
            fftw3.fftw_destroy_plan(self.plan_backward)
            fftw3.fftw_free(self.carrayX)
            fftw3.fftw_free(self.carrayK)
        else:
            IF MPI4PY:
                fftw3mpi.fftw_mpi_cleanup()

    def print_carrayX(self):
        cdef int ii, r
        for r in xrange(self.nb_proc):
            self.comm.barrier()
            sleep(0.05)
            if self.rank == r:
                for ii in xrange(2*self.n_alloc_local):
                    print(self.rank, ii,
                          ' self.carrayX[ii] =', self.carrayX[ii])

    def print_carrayK(self):
        cdef int ii, r
        for r in xrange(self.nb_proc):
            self.comm.barrier()
            sleep(0.05)
            if self.rank == r:
                for ii in xrange(self.n_alloc_local):
                    print(self.rank, ii, ' self.carrayK[ii] =',
                          self.carrayK[ii])

    def describe(self):
        if self.rank == 0:
            print('object of class ', self.__class__,
                  '\nn0 =', self.n0, 'n1 =', self.n1,
                  '\nTRANSPOSED =', self.TRANSPOSED,
                  '\nnb_proc =', self.nb_proc)
            if self.nb_proc == 1:
                print('=> sequenciel version')
            else:
                print('=> parallel version (MPI)')

    def gather_Xspace(self, np.ndarray ff_loc,
                      root=None, type DTYPEf=None):
        cdef np.ndarray ff_seq
        if DTYPEf is None:
            DTYPEf = DTYPEf
        # self.shapeX_loc is the same for all processes,
        # it is safe to use Allgather or Gather
        if root is None:
            ff_seq = np.empty(self.shapeX_seq, DTYPEf)
            self.comm.Allgather(ff_loc, ff_seq)
        elif isinstance(root, int):
            ff_seq = None
            if self.rank == root:
                ff_seq = np.empty(self.shapeX_seq, DTYPEf)
            self.comm.Gather(ff_loc, ff_seq, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_seq

    def scatter_Xspace(self, np.ndarray ff_seq,
                       int root=0, type DTYPEf=None):
        cdef np.ndarray ff_loc
        if DTYPEf is None:
            DTYPEf = DTYPEf
        ff_loc = np.empty(self.shapeX_loc, dtype=DTYPEf)
        # self.shapeX_loc is the same for all processes,
        # it is safe to use Scatter
        if isinstance(root, int):
            self.comm.Scatter(ff_seq, ff_loc, root=root)
        else:
            raise ValueError('root should be an int')
        return ff_loc

    def gather_Kspace(self, np.ndarray ff_fft_loc, root=None):
        cdef np.ndarray ff_fft_seq
        cdef np.ndarray shapeK_temp

        if self.TRANSPOSED is True:
            shapeK_temp = np.empty(2)
            shapeK_temp[0] = self.shapeK_seq[1]
            shapeK_temp[1] = self.shapeK_seq[0]
        else:
            shapeK_temp = self.shapeK_seq
        rowtype = MPI.COMPLEX16.Create_contiguous(self.shapeK_loc[1])
        rowtype.Commit()
        if root is None:
            ff_fft_seq = np.empty(shapeK_temp, dtype=DTYPEc)
            counts = self.comm.allgather(self.shapeK_loc[0])
            self.comm.Allgatherv(sendbuf=[ff_fft_loc, MPI.COMPLEX16],
                                 recvbuf=[ff_fft_seq, (counts, None), rowtype])
            if self.TRANSPOSED is True:
                ff_fft_seq = ff_fft_seq.transpose()
        elif isinstance(root, int):
            ff_fft_seq = None
            if self.rank == root:
                ff_fft_seq = np.empty(shapeK_temp, dtype=DTYPEc)
            counts = self.comm.gather(self.shapeK_loc[0], root=root)
            self.comm.Gatherv(sendbuf=[ff_fft_loc, MPI.COMPLEX16],
                              recvbuf=[ff_fft_seq, (counts, None), rowtype],
                              root=root)
            if self.rank == root and (self.TRANSPOSED is True):
                ff_fft_seq = ff_fft_seq.transpose()
        else:
            raise ValueError('root should be an int')
        rowtype.Free()
        return ff_fft_seq


    # ATTENTION C'EST FAUX !!!!!!
    # C'EST LA VERSION DE CCYLIB !!!!!!
    # IL FAUDRA ECRIRE CA COMME IL FAUT
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

        # NO !!!!!!!!!!

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
        if value == None:
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

    def compute_energy_from_Fourier(self, ff_fft):
        if self.nb_proc > 1:
            raise NotImplementedError('Not yet implemented for mpi')
        return (np.sum(abs(ff_fft[:, 0])**2 + abs(ff_fft[:, -1])**2)
                + 2*np.sum(abs(ff_fft[:, 1:-1])**2))/2

    def compute_energy_from_spatial(self, ff):
        if self.nb_proc > 1:
            raise NotImplementedError('Not yet implemented for mpi')

        return np.mean(abs(ff)**2)/2

    def project_fft_on_realX(self,
                             np.ndarray[DTYPEc_t, ndim=2] f_fft):
        """Project the given field in spectral space such as its
        inverse fft is a real field."""
        cdef np.uint32_t nky_seq
        cdef np.uint32_t iky_ky0, iky_kyM,  ikx_kx0, ikx_kxM,
        cdef np.uint32_t ikyp, ikyn
        cdef DTYPEc_t f_kp_kx0, f_kn_kx0, f_kp_kxM, f_knp_kxM

        if self.nb_proc == 1 or self.SEQUENCIAL:
            nky_seq = self.shapeK_seq[0]

            iky_ky0 = 0
            iky_kyM = nky_seq//2
            ikx_kx0 = 0
            ikx_kxM = self.shapeK_seq[1]-1

            # first, some values have to be real
            f_fft[iky_ky0, ikx_kx0] = f_fft[iky_ky0, ikx_kx0].real
            f_fft[iky_ky0, ikx_kxM] = f_fft[iky_ky0, ikx_kxM].real
            f_fft[iky_kyM, ikx_kx0] = f_fft[iky_kyM, ikx_kx0].real
            f_fft[iky_kyM, ikx_kxM] = f_fft[iky_kyM, ikx_kxM].real

            # second, there are relations between some values
            for ikyp in xrange(1, iky_kyM):
                ikyn = nky_seq - ikyp

                f_kp_kx0 = f_fft[ikyp, ikx_kx0]
                f_kn_kx0 = f_fft[ikyn, ikx_kx0]
                f_fft[ikyp, ikx_kx0] = (f_kp_kx0+f_kn_kx0.conjugate()
                                        )/2
                f_fft[ikyn, ikx_kx0] = ((f_kp_kx0+f_kn_kx0.conjugate()
                                         )/2).conjugate()

                f_kp_kxM = f_fft[ikyp, ikx_kxM]
                f_kn_kxM = f_fft[ikyn, ikx_kxM]
                f_fft[ikyp, ikx_kxM] = (f_kp_kxM+f_kn_kxM.conjugate()
                                        )/2
                f_fft[ikyn, ikx_kxM] = ((f_kp_kxM+f_kn_kxM.conjugate()
                                         )/2).conjugate()
        else:
            nky_seq = self.shapeK_seq[0]
            iky_kyM = nky_seq // 2

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

            if self.rank == self.nb_proc-1:
                ikx_kxM = f_fft.shape[0]-1
                # first, some values have to be real
                f_fft[ikx_kxM, 0] = f_fft[ikx_kxM, 0].real
                f_fft[ikx_kxM, iky_kyM] = f_fft[ikx_kxM, iky_kyM].real
                # second, there are relations between some values
                for ikyp in xrange(1, iky_kyM):
                    ikyn = nky_seq - ikyp
                    f_kp_kxM = f_fft[ikx_kxM, ikyp]
                    f_kn_kxM = f_fft[ikx_kxM, ikyn]
                    f_fft[ikx_kxM, ikyp] = (
                        f_kp_kxM+f_kn_kxM.conjugate())/2
                    f_fft[ikx_kxM, ikyn] = (
                        (f_kp_kxM+f_kn_kxM.conjugate())/2).conjugate()
