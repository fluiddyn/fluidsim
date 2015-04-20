"""
Numerical operators (:mod:`fluidsim.operators.operators`)
===============================================================

.. currentmodule:: fluidsim.operators.operators

This module is written in Cython and provides the classes:

.. autoclass:: Operators
   :members:
   :private-members:

.. autoclass:: GridPseudoSpectral2D
   :members:
   :private-members:

.. autoclass:: OperatorsPseudoSpectral2D
   :members:
   :private-members:


"""

# # DEF MPI4PY = 0

import sys

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

from fluidsim.base.setofvariables import SetOfVariables

from fluidsim.operators.fft import easypyfft

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


cdef class Operators(object):
    pass


cdef class GridPseudoSpectral2D(Operators):
    """Describes a discretisation in spectral and spatial space.

    Parameters
    ----------

    nx, ny : int
        Number of colocation points in the x and y directions

    Lx, Lx : float
        Dimension of the numerical box in the x and y directions

    op_fft2d : :class:`FFT2Dmpi`
        A instance of the class :class:`OP_FFT2Dmpi`

    SEQUENCIAL : bool
        If True, the fft is sequencial even though ``nb_proc > 1``

    """
    # number of nodes, sequenciel case
    cdef public int nx_seq, ny_seq, nkx_seq, nky_seq
    # number of nodes locally stored
    cdef public int nx_loc, ny_loc, nkx_loc, nky_loc
    cdef public DTYPEf_t Lx, Ly, deltax, deltay,
    cdef public DTYPEf_t deltakx, deltaky, deltakh, kmax, kymax

    # shape of the arrays in the physical and Fourier spaces,
    # for the sequential case:
    cdef public np.ndarray shapeX_seq,  shapeK_seq
    # and for the parallel case:
    cdef public np.ndarray shapeX_loc,  shapeK_loc
    # the names without loc or seq correspond to local (or general) quantities
    cdef public np.ndarray shapeX,  shapeK
    # shape K when gathered:
    cdef public np.ndarray shapeK_gat

    cdef public int idimx, idimy, idimkx, idimky

    # these names without loc or seq correspond to local quantities
    cdef public np.ndarray XX, YY, RR, KX, KY, KK, K2, K4, K8, KX2, KY2
    cdef public np.ndarray kx_loc, ky_loc
    cdef public np.ndarray x_seq, y_seq

    # the communicator, nb of processus and rank of the processus
    IF MPI4PY:
        cdef public MPI.Comm comm
    cdef public int nb_proc, rank
    cdef public int iX0loc_start, iKxloc_start, iKyloc_start
    cdef public int nK0_loc, nK1_loc, dim_kx, dim_ky
    cdef public np.ndarray iKxloc_start_rank

    cdef public DTYPEb_t TRANSPOSED, SEQUENCIAL
    cdef public DTYPEb_t SAME_SIZE_IN_ALL_PROC

    # cdef public object where_is_wavenumber

    def __init__(self, int nx, int ny,
                 DTYPEf_t Lx=2*np.pi, DTYPEf_t Ly=2*np.pi,
                 op_fft2d=None, SEQUENCIAL=None):
        if ny % 2 != 0 or nx % 2 != 0:
            raise ValueError('conditions n0 and n1 even not fulfill')

        # n0 is ny and n1 is ny (see def of n0 and n1 in the fftw doc)
        self.nx_seq = int(nx)
        self.ny_seq = int(ny)

        self.Lx = np.float(Lx)
        self.Ly = np.float(Ly)

        self.deltax = self.Lx/self.nx_seq
        self.deltay = self.Ly/self.ny_seq

        self.x_seq = self.deltax * np.arange(self.nx_seq)
        self.y_seq = self.deltay * np.arange(self.ny_seq)

        self.deltakx = 2*np.pi/self.Lx
        self.deltaky = 2*np.pi/self.Ly
        self.deltakh = self.deltakx

        self.nkx_seq = int(self.nx_seq/2.+1)
        self.nky_seq = self.ny_seq

        self.kymax = self.deltaky*self.nky_seq/2.

        # info on MPI
        self.nb_proc = nb_proc
        self.rank = rank
        if self.nb_proc > 1:
            self.comm = comm

        self.shapeX_seq = np.array([self.ny_seq, self.nx_seq])
        self.shapeK_seq = np.array([self.nky_seq, self.nkx_seq])

        if self.nb_proc == 1 or SEQUENCIAL:
            self.SEQUENCIAL = True
            self.SAME_SIZE_IN_ALL_PROC = True
            self.shapeX = self.shapeX_seq
            self.shapeK = self.shapeK_seq
            self.shapeX_loc = self.shapeX_seq
            self.shapeK_loc = self.shapeK_seq
            self.shapeK_gat = self.shapeK_seq

            self.iX0loc_start = 0
            self.iKxloc_start = 0
            self.iKyloc_start = 0

            self.nx_loc = self.nx_seq
            self.ny_loc = self.ny_seq
            self.nkx_loc = self.nkx_seq
            self.nky_loc = self.nky_seq

            self.idimx = 1
            self.idimy = 0
            self.idimkx = 1
            self.idimky = 0

            self.TRANSPOSED = False

        else:
            if nx/2+1 < self.nb_proc:
                raise ValueError('condition nx/2+1 >= nb_proc not fulfill')

            self.SEQUENCIAL = False
            if op_fft2d is None:
                raise ValueError(
                    'for parallel grid, init() needs a op_fft2d object')
            if not hasattr(op_fft2d, 'shapeX_loc'):
                raise ValueError(
                    'The fft operator does not have "shapeX_loc", '
                    'which seems to indicate that it can not run with mpi.')
            self.shapeK_gat = op_fft2d.shapeK_gat
            self.shapeX_loc = op_fft2d.shapeX_loc
            self.shapeK_loc = op_fft2d.shapeK_loc
            self.shapeX = op_fft2d.shapeX_loc
            self.shapeK = op_fft2d.shapeK_loc

            self.idimkx = op_fft2d.idimkx
            self.idimky = op_fft2d.idimky
            self.idimx = op_fft2d.idimx
            self.idimy = op_fft2d.idimy

            self.nx_loc = self.shapeX_loc[self.idimx]
            self.ny_loc = self.shapeX_loc[self.idimy]
            self.nkx_loc = self.shapeK_loc[self.idimkx]
            self.nky_loc = self.shapeK_loc[self.idimky]
            self.iX0loc_start = op_fft2d.iX0loc_start

            self.iKxloc_start = op_fft2d.iKxloc_start
            self.iKyloc_start = op_fft2d.iKyloc_start

            self.iKxloc_start_rank = np.array(
                comm.allgather(self.iKxloc_start))

            nkx_loc_rank = np.array(comm.allgather(self.nkx_loc))
            a = nkx_loc_rank
            self.SAME_SIZE_IN_ALL_PROC = (a >= a.max()).all()

            self.TRANSPOSED = op_fft2d.TRANSPOSED

        self.nK0_loc = self.shapeK_loc[0]
        self.nK1_loc = self.shapeK_loc[1]

        x_loc = self.deltax * np.arange(self.nx_loc)
        y_loc = (self.deltay *
                 np.arange(self.iX0loc_start, self.iX0loc_start+self.ny_loc))
        [self.XX, self.YY] = np.meshgrid(x_loc, y_loc)
        self.RR = np.sqrt((self.XX-self.Lx/2)**2 + (self.YY-self.Ly/2)**2)

        self.kx_loc = self.deltakx * np.arange(self.iKxloc_start,
                                               self.iKxloc_start+self.nkx_loc)
        self.ky_loc = self.deltaky * np.arange(self.iKyloc_start,
                                               self.iKyloc_start+self.nky_loc)
        self.ky_loc[self.ky_loc > self.kymax] = (
            self.ky_loc[self.ky_loc > self.kymax] -
            2*self.kymax)

        if not self.TRANSPOSED:
            [self.KX, self.KY] = np.meshgrid(self.kx_loc, self.ky_loc)
            self.dim_kx = 1
            self.dim_ky = 0
        else:
            [self.KY, self.KX] = np.meshgrid(self.ky_loc, self.kx_loc)
            self.dim_kx = 0
            self.dim_ky = 1

        self.KX2 = self.KX**2
        self.KY2 = self.KY**2
        self.K2 = self.KX2 + self.KY2
        self.K4 = self.K2**2
        self.K8 = self.K4**2
        self.KK = np.sqrt(self.K2)

        self.kmax = np.sqrt((self.deltakx*self.nx_seq)**2 +
                            (self.deltaky*self.ny_seq)**2)/2

    def where_is_wavenumber(self, kx_approx, ky_approx):
        ikx_seq = np.round(kx_approx/self.deltakh)

        if ikx_seq >= self.nkx_seq:
            raise ValueError('not good :-) ikx_seq >= self.nkx_seq')

        if self.SEQUENCIAL:
            rank_k = 0
            ikx_loc = ikx_seq
        else:
            if self.SAME_SIZE_IN_ALL_PROC:
                rank_k = int(np.floor(float(ikx_seq)/self.nkx_loc))
            else:
                rank_k = 0
                while (rank_k < self.nb_proc-1 and
                       (not (self.iKxloc_start_rank[rank_k] <= ikx_seq and
                             ikx_seq < self.iKxloc_start_rank[rank_k+1]))):
                    rank_k += 1

            ikx_loc = ikx_seq - self.iKxloc_start_rank[rank_k]

        iky_loc = np.round(ky_approx/self.deltaky)
        if iky_loc < 0:
            iky_loc = self.nky_loc+iky_loc

        if self.TRANSPOSED:
            ik0_loc = ikx_loc
            ik1_loc = iky_loc
        else:
            ik0_loc = iky_loc
            ik1_loc = ikx_loc

        return rank_k, ik0_loc, ik1_loc


cdef class OperatorsPseudoSpectral2D(GridPseudoSpectral2D):
    """Provides fast Fourier transform functions and 2D operators.

    `type_fft='FFTWCY'` :
    cython wrapper of plans
    fftw_plan_dft_r2c_2d / fftw_plan_dft_c2r_2d (sequencial case)
    and
    fftw_mpi_plan_dft_r2c_2d / fftw_mpi_plan_dft_c2r_2d
    (parallel case)

    `type_fft='FFTWCCY'` :
    cython wrapper of a self-written c libray using
    sequencial fftw plans and MPI_Type. Seems to be faster than
    the implementation of the mpi FFT by fftw (lib fftw-mpi).

    `type_fft='FFTWPY'` :
    use of the module :mod:`easypyfft2D` with fftw

    `type_fft='FFTPACK'
    use of the module :mod:`easypyfft2D` with fftp
    (bad and slow implementation!)

    """

    cdef public DTYPEf_t coef_dealiasing
    cdef public object fft2, ifft2
    cdef public object gather_Xspace,  gather_Kspace,
    cdef public object scatter_Xspace, scatter_Kspace
    cdef public object project_fft_on_realX
    cdef public object params
    cdef public np.ndarray K2_not0, K4_not0, KX_over_K2, KY_over_K2
    cdef public np.ndarray Kappa2, Kappa_over_ic, f_over_c2Kappa2
    cdef public np.ndarray where_dealiased

    cdef public int nkxE, nkyE, nkhE
    cdef public np.ndarray kxE, kyE, khE

    cdef public str type_fft

    @staticmethod
    def _complete_params_with_default(params):
        """This static method is used to complete the *params* container.
        """
        if nb_proc > 1:
            type_fft = 'FFTWCCY'
        else:
            if not sys.platform == 'win32':
                type_fft = 'FFTWCY'
            else:
                type_fft = 'FFTWPY'

        attribs = {'type_fft': type_fft,
                   'TRANSPOSED_OK': True,
                   'coef_dealiasing': 2./3,
                   'nx': 48,
                   'ny': 48,
                   'Lx': 8,
                   'Ly': 8}
        params._set_child('oper', attribs=attribs)

    def __init__(self,
                 SEQUENCIAL=None,
                 params=None,
                 goal_to_print=None):

        if not params.ONLY_COARSE_OPER:
            nx = int(params.oper.nx)
            ny = int(params.oper.ny)
        else:
            nx = 4
            ny = 4

        Lx = params.oper.Lx
        Ly = params.oper.Ly
        type_fft = str(params.oper.type_fft)
        coef_dealiasing = params.oper.coef_dealiasing
        TRANSPOSED = params.oper.TRANSPOSED_OK

        # if rank == 0:
        #     to_print = 'Init. operator'
        #     if goal_to_print is not None:
        #         to_print += ' ('+goal_to_print+')'
        #     print(to_print)

        if params is not None:
            self.params = params

        list_type_fft = ['FFTWCY', 'FFTWCCY', 'FFTWPY', 'FFTPACK']
        if type_fft not in list_type_fft:
            raise ValueError('type_fft should be in ' + repr(list_type_fft))

        if type_fft == 'FFTWCCY' and nb_proc == 1:
            type_fft = 'FFTWCY'

        try:
            if type_fft == 'FFTWCY':
                import fluidsim.operators.fft.fftw2dmpicy as fftw2Dmpi
        except ImportError as err:
            if nb_proc == 1:
                type_fft = 'FFTWPY'
            else:
                type_fft = 'FFTWCCY'

        try:
            if type_fft == 'FFTWCCY':
                # We need to do this check because after the first
                # import, the import statement doesn't raise the
                # ImportError correctly (is it normal?)
                if nb_proc == 1:
                    raise ImportError(
                        'fftw2Dmpiccy only works if MPI.COMM_WORLD.size > 1.')
                import fluidsim.operators.fft.fftw2dmpiccy as fftw2Dmpi
        except ImportError:
            if nb_proc > 1 and SEQUENCIAL is None:
                raise ValueError(
                    'if nb_proc>1, we need one of the libraries '
                    'fftw2Dmpicy or fftw2Dmpiccy')
            type_fft = 'FFTWPY'

        if type_fft == 'FFTWPY':
            try:
                import pyfftw
            except ImportError as err:
                type_fft = 'FFTPACK'

        self.type_fft = type_fft
        
        # Initialization of the fft transforms
        if type_fft not in ['FFTWPY', 'FFTPACK']:
            if not TRANSPOSED and type_fft == 'FFTWCCY':
                raise ValueError('FFTWCCY does not suport the '
                                 '(inefficient!) option TRANSPOSED=False')

            if type_fft == 'FFTWCY':
                op_fft2d = fftw2Dmpi.FFT2Dmpi(ny, nx,
                                              TRANSPOSED=TRANSPOSED,
                                              SEQUENCIAL=SEQUENCIAL)
            else:
                op_fft2d = fftw2Dmpi.FFT2Dmpi(ny, nx)
            if op_fft2d.nb_proc > 1:
                self.gather_Xspace = op_fft2d.gather_Xspace
                self.gather_Kspace = op_fft2d.gather_Kspace
                self.scatter_Xspace = op_fft2d.scatter_Xspace
                self.scatter_Kspace = op_fft2d.scatter_Kspace

        elif type_fft == 'FFTWPY':
            op_fft2d = easypyfft.FFTW2DReal2Complex(nx, ny)
        elif type_fft == 'FFTPACK':
            op_fft2d = easypyfft.fftp2D(nx, ny)

        self.fft2 = op_fft2d.fft2d
        self.ifft2 = op_fft2d.ifft2d

        GridPseudoSpectral2D.__init__(self, nx, ny, Lx, Ly,
                                      op_fft2d=op_fft2d, SEQUENCIAL=SEQUENCIAL)

        self.K2_not0 = self.K2.copy()
        self.K4_not0 = self.K4.copy()
        if rank == 0 or SEQUENCIAL:
            self.K2_not0[0, 0] = 10.e-10
            self.K4_not0[0, 0] = 10.e-10

        self.KX_over_K2 = self.KX/self.K2_not0
        self.KY_over_K2 = self.KY/self.K2_not0

        try:
            self.Kappa2 = self.K2 + self.params.kd2

            self.Kappa_over_ic = -1.j*np.sqrt(
                self.Kappa2/self.params.c2
                )

            if self.params.f != 0:
                self.f_over_c2Kappa2 = self.params.f/(
                    self.params.c2*self.Kappa2
                    )

        except AttributeError:
            pass

        # for spectra, we forget the larger wavenumber,
        # since there is no energy inside because of dealiasing
        self.nkxE = self.nkx_seq - 1
        self.nkyE = self.nky_seq/2

        self.kxE = self.deltakx * np.arange(self.nkxE)
        self.kyE = self.deltaky * np.arange(self.nkyE)
        self.khE = self.kxE
        self.nkhE = self.nkxE

        # Initialisation dealiasing
        self.coef_dealiasing = coef_dealiasing
        CONDKX = abs(self.KX) > self.coef_dealiasing*self.kxE.max()
        CONDKY = abs(self.KY) > self.coef_dealiasing*self.kyE.max()
        where_dealiased = np.logical_or(CONDKX, CONDKY)

        self.where_dealiased = np.array(where_dealiased, dtype=DTYPEb)

        try:
            self.project_fft_on_realX = op_fft2d.project_fft_on_realX
        except KeyError:
            if nb_proc > 1:
                raise ValueError(
                    'nb_proc > 1 but no function'
                    'project_fft_on_realX defined')
            self.project_fft_on_realX = self.project_fft_on_realX_seq

    def produce_str_describing_oper(self):
        """Produce a string describing the operator."""
        if (self.Lx/np.pi).is_integer():
            str_Lx = repr(int(self.Lx/np.pi)) + 'pi'
        else:
            str_Lx = '{:.3f}'.format(self.Lx).rstrip('0')
        if (self.Ly/np.pi).is_integer():
            str_Ly = repr(int(self.Ly/np.pi)) + 'pi'
        else:
            str_Ly = '{:.3f}'.format(self.Ly).rstrip('0')
        return ('L='+str_Lx+'x'+str_Ly+'_{}x{}').format(
            self.nx_seq, self.ny_seq)

    def produce_long_str_describing_oper(self):
        """Produce a string describing the operator."""
        if (self.Lx/np.pi).is_integer():
            str_Lx = repr(int(self.Lx/np.pi)) + '*pi'
        else:
            str_Lx = '{:.3f}'.format(self.Lx).rstrip('0')
        if (self.Ly/np.pi).is_integer():
            str_Ly = repr(int(self.Ly/np.pi)) + '*pi'
        else:
            str_Ly = '{:.3f}'.format(self.Ly).rstrip('0')
        return (
            'type fft: ' + self.type_fft + '\n' +
            'nx = {0:6d} ; ny = {1:6d}\n'.format(self.nx_seq, self.ny_seq) +
            'Lx = ' + str_Lx + ' ; Ly = ' + str_Ly + '\n')

    # def rotfft_from_vecfft(self, vecx_fft, vecy_fft):
    #     """Return the rotational (curl) of a vector in spectral space."""
    #     return 1j*( self.KX*vecy_fft - self.KY*vecx_fft )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def rotfft_from_vecfft(self,
                           np.ndarray[DTYPEc_t, ndim=2] vecx_fft,
                           np.ndarray[DTYPEc_t, ndim=2] vecy_fft):
        """Return the rotational of a vector in spectral space."""
        cdef Py_ssize_t i0, i1, n0, n1
        cdef np.ndarray[DTYPEc_t, ndim=2] rot_fft
        cdef np.ndarray[DTYPEf_t, ndim=2] KX, KY

        n0 = self.nK0_loc
        n1 = self.nK1_loc

        KX = self.KX
        KY = self.KY
        rot_fft = np.empty([n0, n1], dtype=np.complex128)

        for i0 in range(n0):
            for i1 in range(n1):
                rot_fft[i0, i1] = 1j*(KX[i0, i1]*vecy_fft[i0, i1]
                                      - KY[i0, i1]*vecx_fft[i0, i1])
        return rot_fft

    # def divfft_from_vecfft_old(self, vecx_fft, vecy_fft):
    #     """Return the divergence of a vector in spectral space."""
    #     return 1j*( self.KX*vecx_fft + self.KY*vecy_fft )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def divfft_from_vecfft(self,
                           np.ndarray[DTYPEc_t, ndim=2] vecx_fft,
                           np.ndarray[DTYPEc_t, ndim=2] vecy_fft):
        """Return the divergence of a vector in spectral space."""
        cdef Py_ssize_t i0, i1, n0, n1
        cdef np.ndarray[DTYPEc_t, ndim=2] div_fft
        cdef np.ndarray[DTYPEf_t, ndim=2] KX, KY

        n0 = self.nK0_loc
        n1 = self.nK1_loc

        KX = self.KX
        KY = self.KY
        div_fft = np.empty([n0, n1], dtype=np.complex128)

        for i0 in xrange(n0):
            for i1 in xrange(n1):
                div_fft[i0, i1] = 1j*(KX[i0, i1]*vecx_fft[i0, i1]
                                      + KY[i0, i1]*vecy_fft[i0, i1]
                                      )
        return div_fft

    def vecfft_from_rotfft(self, rot_fft):
        """Return the velocity in spectral space computed from the
        rotational."""
        ux_fft = 1j * self.KY_over_K2*rot_fft
        uy_fft = -1j * self.KX_over_K2*rot_fft
        return ux_fft, uy_fft

    def vecfft_from_divfft(self, div_fft):
        """Return the velocity in spectral space computed from the
        divergence."""
        ux_fft = -1j * self.KX_over_K2*div_fft
        uy_fft = -1j * self.KY_over_K2*div_fft
        return ux_fft, uy_fft




    # def gradfft_from_fft_old(self, f_fft):
    #     """Return the gradient of f_fft in spectral space."""
    #     px_f_fft = 1j * self.KX*f_fft
    #     py_f_fft = 1j * self.KY*f_fft
    #     return px_f_fft, py_f_fft


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def pxffft_from_fft(self, f_fft):
        """Return the gradient of f_fft in spectral space."""
        cdef Py_ssize_t i0, i1, n0, n1
        cdef np.ndarray[DTYPEf_t, ndim=2] KX
        cdef np.ndarray[DTYPEc_t, ndim=2] px_f_fft

        cdef np.ndarray[DTYPEc_t, ndim=2] fc_fft
        cdef np.ndarray[DTYPEf_t, ndim=2] ff_fft

        n0 = self.nK0_loc
        n1 = self.nK1_loc

        KX = self.KX
        KY = self.KY

        px_f_fft = np.empty([n0, n1], dtype=np.complex128)

        if f_fft.dtype == np.float64:
            ff_fft = f_fft
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    px_f_fft[i0, i1] = 1j * KX[i0, i1]*ff_fft[i0, i1]
        else:
            fc_fft = f_fft
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    px_f_fft[i0, i1] = 1j * KX[i0, i1]*fc_fft[i0, i1]

        return px_f_fft

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def gradfft_from_fft(self, f_fft):
        """Return the gradient of f_fft in spectral space."""
        cdef Py_ssize_t i0, i1, n0, n1
        cdef np.ndarray[DTYPEf_t, ndim=2] KX, KY
        cdef np.ndarray[DTYPEc_t, ndim=2] px_f_fft, py_f_fft

        cdef np.ndarray[DTYPEc_t, ndim=2] fc_fft
        cdef np.ndarray[DTYPEf_t, ndim=2] ff_fft

        n0 = self.nK0_loc
        n1 = self.nK1_loc

        KX = self.KX
        KY = self.KY

        px_f_fft = np.empty([n0, n1], dtype=np.complex128)
        py_f_fft = np.empty([n0, n1], dtype=np.complex128)

        if f_fft.dtype == np.float64:
            ff_fft = f_fft
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    px_f_fft[i0, i1] = 1j * KX[i0, i1]*ff_fft[i0, i1]
                    py_f_fft[i0, i1] = 1j * KY[i0, i1]*ff_fft[i0, i1]
        else:
            fc_fft = f_fft
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    px_f_fft[i0, i1] = 1j * KX[i0, i1]*fc_fft[i0, i1]
                    py_f_fft[i0, i1] = 1j * KY[i0, i1]*fc_fft[i0, i1]

        return px_f_fft, py_f_fft

    def projection_perp(self, fx_fft, fy_fft):
        KX = self.KX
        KY = self.KY
        a = fx_fft - self.KX_over_K2*(KX*fx_fft+KY*fy_fft)
        b = fy_fft - self.KY_over_K2*(KX*fx_fft+KY*fy_fft)
        fx_fft[:] = a
        fy_fft[:] = b
        return a, b

    def uxuyfft_from_psifft(self, psi_fft):
        px_psi_fft, py_psi_fft = self.gradfft_from_fft(psi_fft)
        ux_fft = -py_psi_fft
        uy_fft = px_psi_fft
        return ux_fft, uy_fft

    def rotfft_from_psifft(self, psi_fft):
        rot_fft = -self.K2*psi_fft
        return rot_fft

    def uxuyetafft_from_qfft(self, q_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
        K2 = self.K2
        K2_not0 = self.K2_not0
        rot_fft = K2*q_fft/(K2_not0+params.kd2)
        if rank == 0:
            rot_fft[0, 0] = 0.
        ux_fft, uy_fft = self.vecfft_from_rotfft(rot_fft)

        if params.f == 0:
            eta_fft = self.constant_arrayK(value=0)
        else:
            eta_fft = -params.f*q_fft/(K2_not0+params.kd2)/params.c2
        if rank == 0:
            eta_fft[0, 0] = 0.
        return ux_fft, uy_fft, eta_fft

    def uxuyetafft_from_afft(self, a_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
        K2 = self.K2
        K2_not0 = self.K2_not0

        if params.f == 0:
            rot_fft = self.constant_arrayK(value=0)
        else:
            rot_fft = params.f*a_fft/(K2_not0+params.kd2)
        if rank == 0:
            rot_fft[0, 0] = 0.
        ux_fft, uy_fft = self.vecfft_from_rotfft(rot_fft)

        eta_fft = a_fft/(K2_not0+params.kd2)
        if rank == 0:
            eta_fft[0, 0] = 0.
        return ux_fft, uy_fft, eta_fft

    def rotfft_from_qfft(self, q_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
        K2 = self.K2
        K2_not0 = self.K2_not0
        rot_fft = K2*q_fft/(K2_not0+params.kd2)
        if rank == 0:
            rot_fft[0, 0] = 0.
        return rot_fft

    def rotfft_from_afft(self, a_fft, params=None):
        """Compute ux, uy and eta in Fourier space."""
        if params is None:
            params = self.params
        K2 = self.K2
        K2_not0 = self.K2_not0
        if params.f == 0:
            rot_fft = self.constant_arrayK(value=0)
        else:
            rot_fft = params.f*a_fft/(K2_not0+params.kd2)
        if rank == 0:
            rot_fft[0, 0] = 0.
        return rot_fft

    def afft_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft,
                             params=None):
        if params is None:
            params = self.params
        rot_fft = self.rotfft_from_vecfft(ux_fft, uy_fft)
        a_fft = self.K2*eta_fft
        if params.f != 0:
            a_fft += params.f/params.c2*rot_fft
        return a_fft

    def etafft_from_qfft(self, q_fft, params=None):
        """Compute eta in Fourier space."""
        if params is None:
            params = self.params
        K2_not0 = self.K2_not0
        if params.f == 0:
            eta_fft = self.constant_arrayK(value=0)
        else:
            eta_fft = -params.f/params.c2*q_fft/(K2_not0+params.kd2)
        if rank == 0:
            eta_fft[0, 0] = 0.
        return eta_fft

    def etafft_from_afft(self, a_fft, params=None):
        """Compute eta in Fourier space."""
        if params is None:
            params = self.params
        K2_not0 = self.K2_not0
        eta_fft = a_fft/(K2_not0+params.kd2)
        if rank == 0:
            eta_fft[0, 0] = 0.
        return eta_fft

    def etafft_from_aqfft(self, a_fft, q_fft, params=None):
        """Compute eta in Fourier space."""
        if params is None:
            params = self.params
        K2_not0 = self.K2_not0
        if params.f == 0:
            eta_fft = a_fft/K2_not0
        else:
            eta_fft = (a_fft - params.f/params.c2*q_fft)/(
                K2_not0+params.kd2)
        if rank == 0:
            eta_fft[0, 0] = 0.
        return eta_fft

    def qdafft_from_uxuyetafft(self, ux_fft, uy_fft, eta_fft, params=None):
        if params is None:
            params = self.params
        div_fft = self.divfft_from_vecfft(ux_fft, uy_fft)
        rot_fft = self.rotfft_from_vecfft(ux_fft, uy_fft)
        q_fft = rot_fft - params.f*eta_fft
        ageo_fft = params.f/params.c2*rot_fft + self.K2*eta_fft
        return q_fft, div_fft, ageo_fft

    def apamfft_from_adfft(self, a_fft, d_fft):
        """Return the engein modes ap and am."""
        Delta_a_fft = self.Kappa_over_ic*d_fft
        ap_fft = 0.5*(a_fft + Delta_a_fft)
        am_fft = 0.5*(a_fft - Delta_a_fft)
        return ap_fft, am_fft

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def divfft_from_apamfft(self, ap_fft, am_fft):
        """Return div from the engein modes ap and am."""
        cdef Py_ssize_t i0, i1, n0, n1
        cdef Py_ssize_t rank = self.rank
        cdef np.ndarray[DTYPEc_t, ndim=2] Kappa_over_ic, Delta_a_fft
        cdef np.ndarray[DTYPEc_t, ndim=2] d_fft

        Delta_a_fft = ap_fft - am_fft
        n0 = self.nK0_loc
        n1 = self.nK1_loc
        Kappa_over_ic = self.Kappa_over_ic
        d_fft = np.empty([n0, n1], dtype=np.complex128)

        for i0 in range(n0):
            for i1 in range(n1):
                if i0 == 0 and i1 == 0 and rank == 0:
                    d_fft[i0, i1] = 0.
                else:
                    d_fft[i0, i1] = (
                        Delta_a_fft[i0, i1]/Kappa_over_ic[i0, i1]
                        )
        return d_fft

    def qapamfft_from_uxuyetafft_old(self, ux_fft, uy_fft, eta_fft,
                                     params=None):
        """ux, uy, eta (fft) ---> q, ap, am (fft)"""
        if params is None:
            params = self.params
        div_fft = self.divfft_from_vecfft(ux_fft, uy_fft)
        rot_fft = self.rotfft_from_vecfft(ux_fft, uy_fft)
        q_fft = rot_fft - params.f*eta_fft
        a_fft = (self.K2*eta_fft
                 + params.f/params.c2*rot_fft)
        ap_fft, am_fft = self.apamfft_from_adfft(a_fft, div_fft)
        if rank == 0:
            ap_fft[0, 0] = ux_fft[0, 0] + 1.j*uy_fft[0, 0]
            am_fft[0, 0] = ux_fft[0, 0] - 1.j*uy_fft[0, 0]
        return q_fft, ap_fft, am_fft

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def qapamfft_from_uxuyetafft(self,
                                 np.ndarray[DTYPEc_t, ndim=2] ux_fft,
                                 np.ndarray[DTYPEc_t, ndim=2] uy_fft,
                                 np.ndarray[DTYPEc_t, ndim=2] eta_fft,
                                 params=None):
        """ux, uy, eta (fft) ---> q, ap, am (fft)"""
        cdef Py_ssize_t i0, i1, n0, n1
        cdef Py_ssize_t rank = self.rank
        cdef np.ndarray[DTYPEf_t, ndim=2] KX, KY, K2
        cdef np.ndarray[DTYPEc_t, ndim=2] Kappa_over_ic
        cdef np.ndarray[DTYPEc_t, ndim=2] q_fft, ap_fft, am_fft
        cdef DTYPEc_t rot_fft, a_over2_fft, Deltaa_over2_fft
        cdef DTYPEf_t freq_Corio, f_over_c2

        if params is None:
            params = self.params

        n0 = self.nK0_loc
        n1 = self.nK1_loc

        KX = self.KX
        KY = self.KY
        K2 = self.K2
        Kappa_over_ic = self.Kappa_over_ic
        KX_over_K2 = self.KX_over_K2
        KY_over_K2 = self.KY_over_K2

        q_fft = np.empty([n0, n1], dtype=np.complex128)
        ap_fft = np.empty([n0, n1], dtype=np.complex128)
        am_fft = np.empty([n0, n1], dtype=np.complex128)

        freq_Corio = params.f
        f_over_c2 = freq_Corio/params.c2

        if freq_Corio != 0:
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    if i0 == 0 and i1 == 0 and rank == 0:
                        q_fft[i0, i1] = 0
                        ap_fft[i0, i1] = ux_fft[0, 0] + 1.j*uy_fft[0, 0]
                        am_fft[i0, i1] = ux_fft[0, 0] - 1.j*uy_fft[0, 0]
                    else:

                        rot_fft = 1j*(
                            KX[i0, i1]*uy_fft[i0, i1] -
                            KY[i0, i1]*ux_fft[i0, i1])

                        q_fft[i0, i1] = rot_fft - freq_Corio*eta_fft[i0, i1]

                        a_over2_fft = 0.5*(
                            K2[i0, i1] * eta_fft[i0, i1] +
                            f_over_c2*rot_fft)

                        Deltaa_over2_fft = 0.5j*Kappa_over_ic[i0, i1]*(
                            KX[i0, i1]*ux_fft[i0, i1] +
                            KY[i0, i1]*uy_fft[i0, i1])

                        ap_fft[i0, i1] = a_over2_fft + Deltaa_over2_fft
                        am_fft[i0, i1] = a_over2_fft - Deltaa_over2_fft

        else:  # (freq_Corio == 0.)
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    if i0 == 0 and i1 == 0 and rank == 0:
                        q_fft[i0, i1] = 0
                        ap_fft[i0, i1] = ux_fft[0, 0] + 1.j*uy_fft[0, 0]
                        am_fft[i0, i1] = ux_fft[0, 0] - 1.j*uy_fft[0, 0]
                    else:
                        q_fft[i0, i1] = 1j*(
                            KX[i0, i1]*uy_fft[i0, i1]
                            - KY[i0, i1]*ux_fft[i0, i1])

                        a_over2_fft = 0.5*K2[i0, i1]*eta_fft[i0, i1]

                        Deltaa_over2_fft = 0.5j*Kappa_over_ic[i0, i1]*(
                            KX[i0, i1]*ux_fft[i0, i1]
                            + KY[i0, i1]*uy_fft[i0, i1])

                        ap_fft[i0, i1] = a_over2_fft + Deltaa_over2_fft
                        am_fft[i0, i1] = a_over2_fft - Deltaa_over2_fft

        return q_fft, ap_fft, am_fft

    def uxuyetafft_from_qapamfft_old(self, q_fft, ap_fft, am_fft):
        """q, ap, am (fft) ---> ux, uy, eta (fft)"""
        a_fft = ap_fft + am_fft
        if rank == 0:
            a_fft[0, 0] = 0.
        div_fft = self.divfft_from_apamfft(ap_fft, am_fft)
        (uxa_fft, uya_fft, etaa_fft
         ) = self.uxuyetafft_from_afft(a_fft)
        (uxq_fft, uyq_fft, etaq_fft
         ) = self.uxuyetafft_from_qfft(q_fft)
        uxd_fft, uyd_fft = self.vecfft_from_divfft(div_fft)
        ux_fft = uxa_fft + uxq_fft + uxd_fft
        uy_fft = uya_fft + uyq_fft + uyd_fft
        eta_fft = etaa_fft + etaq_fft
        if rank == 0:
            ux_fft[0, 0] = 0.5 * (ap_fft[0, 0] + am_fft[0, 0])
            uy_fft[0, 0] = 0.5j * (am_fft[0, 0] - ap_fft[0, 0])
        return ux_fft, uy_fft, eta_fft

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    def uxuyetafft_from_qapamfft(self,
                                 np.ndarray[DTYPEc_t, ndim=2] q_fft,
                                 np.ndarray[DTYPEc_t, ndim=2] ap_fft,
                                 np.ndarray[DTYPEc_t, ndim=2] am_fft,
                                 params=None):
        """q, ap, am (fft) ---> ux, uy, eta (fft)"""
        cdef Py_ssize_t i0, i1, n0, n1
        cdef Py_ssize_t rank = self.rank
        cdef np.ndarray[DTYPEf_t, ndim=2] KX, KY, K2
        cdef np.ndarray[DTYPEc_t, ndim=2] Kappa_over_ic
        cdef np.ndarray[DTYPEf_t, ndim=2] Kappa2
        cdef np.ndarray[DTYPEf_t, ndim=2] f_over_c2Kappa2
        cdef np.ndarray[DTYPEf_t, ndim=2] KX_over_K2, KY_over_K2
        cdef np.ndarray[DTYPEc_t, ndim=2] eta_fft, ux_fft, uy_fft
        cdef DTYPEc_t div_fft, rot_fft
        cdef DTYPEf_t freq_Corio

        if params is None:
            params = self.params

        n0 = self.nK0_loc
        n1 = self.nK1_loc

        KX = self.KX
        KY = self.KY
        K2 = self.K2
        Kappa2 = self.Kappa2
        Kappa_over_ic = self.Kappa_over_ic
        f_over_c2Kappa2 = self.f_over_c2Kappa2
        KX_over_K2 = self.KX_over_K2
        KY_over_K2 = self.KY_over_K2

        eta_fft = np.empty([n0, n1], dtype=np.complex128)
        ux_fft = np.empty([n0, n1], dtype=np.complex128)
        uy_fft = np.empty([n0, n1], dtype=np.complex128)

        freq_Corio = params.f

        if freq_Corio != 0:
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    if rank == 0 and i0 == 0 and i1 == 0:
                        eta_fft[i0, i1] = 0
                        ux_fft[i0, i1] = 0.5 * (ap_fft[0, 0] + am_fft[0, 0])
                        uy_fft[i0, i1] = 0.5j * (am_fft[0, 0] - ap_fft[0, 0])
                    else:
                        div_fft = (
                            ap_fft[i0, i1] - am_fft[i0, i1]
                            )/Kappa_over_ic[i0, i1]
                        eta_fft[i0, i1] = (
                            (ap_fft[i0, i1] + am_fft[i0, i1])/Kappa2[i0, i1]
                            - f_over_c2Kappa2[i0, i1]*q_fft[i0, i1])
                        rot_fft = (
                            q_fft[i0, i1]
                            + freq_Corio*eta_fft[i0, i1])
                        ux_fft[i0, i1] = (
                            1j * KY_over_K2[i0, i1]*rot_fft
                            - 1j * KX_over_K2[i0, i1]*div_fft)
                        uy_fft[i0, i1] = (
                            -1j * KX_over_K2[i0, i1]*rot_fft
                            - 1j * KY_over_K2[i0, i1]*div_fft)

        else:  # (freq_Corio == 0.)
            for i0 in xrange(n0):
                for i1 in xrange(n1):
                    if i0 == 0 and i1 == 0 and rank == 0:
                        eta_fft[i0, i1] = 0
                        ux_fft[i0, i1] = 0.5 * (ap_fft[0, 0] + am_fft[0, 0])
                        uy_fft[i0, i1] = 0.5j * (am_fft[0, 0] - ap_fft[0, 0])
                    else:
                        div_fft = (
                            ap_fft[i0, i1] - am_fft[i0, i1]
                            )/Kappa_over_ic[i0, i1]
                        eta_fft[i0, i1] = (
                            ap_fft[i0, i1] + am_fft[i0, i1]
                            )/K2[i0, i1]
                        rot_fft = q_fft[i0, i1]
                        ux_fft[i0, i1] = (
                            1j*KY_over_K2[i0, i1]*rot_fft
                            - 1j*KX_over_K2[i0, i1]*div_fft)
                        uy_fft[i0, i1] = (
                            - 1j*KX_over_K2[i0, i1]*rot_fft
                            - 1j*KY_over_K2[i0, i1]*div_fft)

        return ux_fft, uy_fft, eta_fft

    def dealiasing(self, *arguments):
        for ii in range(len(arguments)):
            thing = arguments[ii]
            if isinstance(thing, SetOfVariables):
                dealiasing_setofvar(thing, self.where_dealiased,
                                    self.nK0_loc, self.nK1_loc)
            elif isinstance(thing, np.ndarray):
                dealiasing_variable(thing, self.where_dealiased,
                                    self.nK0_loc, self.nK1_loc)

    def dealiasing_setofvar(self, sov):
        dealiasing_setofvar(sov, self.where_dealiased,
                            self.nK0_loc, self.nK1_loc)

    # def sum_wavenumbers_old(self, field_fft):
    #     S_allkx = np.sum(field_fft)
    #     if not self.TRANSPOSED:
    #         S_kx0 = np.sum( field_fft[:,0] )
    #     else:
    #         if self.rank==0:
    #             S_kx0 = np.sum( field_fft[0,:] )
    #         else:
    #             S_kx0 = 0.
    #     S_result = 2*S_allkx-S_kx0
    #     if self.nb_proc>1:
    #         S_result = self.comm.allreduce(S_result, op=MPI.SUM)
    #     return S_result

    def mean_space(self, field):

        mean_field = np.mean(field)
        if not self.SEQUENCIAL:
            mean_field = self.comm.allreduce(mean_field, op=MPI.SUM)
            mean_field /= nb_proc
        return mean_field

    def sum_wavenumbers(self, np.ndarray[DTYPEf_t, ndim=2] A_fft):
        """Sum the given array over all wavenumbers."""
        cdef np.uint32_t ikO, ik1
        cdef np.uint32_t nk0loc, nk1loc, rank, TRANSPOSED
        cdef DTYPEf_t A0D, sum_A_fft

        nk0loc = self.shapeK_loc[0]
        nk1loc = self.shapeK_loc[1]

        rank = self.rank

        if self.TRANSPOSED:
            TRANSPOSED = 1
        else:
            TRANSPOSED = 0

        sum_A_fft = 0.

        for ik0 in range(nk0loc):
            for ik1 in range(nk1loc):
                A0D = A_fft[ik0, ik1]
                if TRANSPOSED == 0:
                    if ik1 > 0:
                        A0D = A0D*2
                else:
                    if ik0 > 0 or rank > 0:
                        A0D = A0D*2
                sum_A_fft += A0D

        # if self.nb_proc>1:
        if not self.SEQUENCIAL:
            sum_A_fft = self.comm.allreduce(sum_A_fft, op=MPI.SUM)
        return sum_A_fft

    def spectra1D_from_fft(self, energy_fft):
        """Compute the 1D spectra. Return a dictionary."""
        if self.nb_proc == 1:
            # In this case, self.dim_ky==0 and self.dim_ky==1
            # Memory is not shared
            # note that only the kx>=0 are in the spectral variables
            # to obtain the spectrum as a function of kx
            # we sum over all ky
            # the 2 is here because there are only the kx>=0
            E_kx = 2.*energy_fft.sum(self.dim_ky)/self.deltakx
            E_kx[0] = E_kx[0]/2
            E_kx = E_kx[:self.nkxE]
            # computation of E_ky
            E_ky_temp = energy_fft[:, 0].copy()
            E_ky_temp += 2*energy_fft[:, 1:].sum(1)
            nkyE = self.nkyE
            E_ky = E_ky_temp[0:nkyE]
            E_ky[1:nkyE] = E_ky[1:nkyE] + E_ky_temp[self.nky_seq:nkyE:-1]
            E_ky = E_ky/self.deltaky

        elif self.TRANSPOSED:
            # In this case, self.dim_ky==1 and self.dim_ky==0
            # Memory is shared along kx
            # note that only the kx>=0 are in the spectral variables
            # to obtain the spectrum as a function of kx
            # we sum er.mamover all ky
            # the 2 is here because there are only the kx>=0
            E_kx_loc = 2.*energy_fft.sum(self.dim_ky)/self.deltakx
            if self.rank == 0:
                E_kx_loc[0] = E_kx_loc[0]/2
            E_kx = np.empty(self.nkxE)
            counts = self.comm.allgather(self.nkx_loc)
            self.comm.Allgatherv(sendbuf=[E_kx_loc, MPI.DOUBLE],
                                 recvbuf=[E_kx, (counts, None), MPI.DOUBLE])
            E_kx = E_kx[:self.nkxE]
            # computation of E_ky
            if self.rank == 0:
                E_ky_temp = energy_fft[0, :]+2*energy_fft[1:, :].sum(0)
            else:
                E_ky_temp = 2*energy_fft.sum(0)
            nkyE = self.nkyE
            E_ky = E_ky_temp[0:nkyE]
            E_ky[1:nkyE] = E_ky[1:nkyE] + E_ky_temp[self.nky_seq:nkyE:-1]
            E_ky = E_ky/self.deltaky
            E_ky = self.comm.allreduce(E_ky, op=MPI.SUM)

        elif not self.TRANSPOSED:
            # In this case, self.dim_ky==0 and self.dim_ky==1
            # Memory is shared along ky
            # note that only the kx>=0 are in the spectral variables
            # to obtain the spectrum as a function of kx
            # we sum over all ky
            # the 2 is here because there are only the kx>=0
            E_kx = 2.*energy_fft.sum(self.dim_ky)/self.deltakx
            E_kx[0] = E_kx[0]/2
            E_kx = self.comm.allreduce(E_kx, op=MPI.SUM)
            E_kx = E_kx[:self.nkxE]
            # computation of E_ky
            E_ky_temp = energy_fft[:, 0].copy()
            E_ky_temp += 2*energy_fft[:, 1:].sum(1)
            E_ky_temp = np.ascontiguousarray(E_ky_temp)
#            print self.rank, 'E_ky_temp', E_ky_temp, E_ky_temp.shape
            E_ky_long = np.empty(self.nky_seq)
            counts = self.comm.allgather(self.nky_loc)
            self.comm.Allgatherv(sendbuf=[E_ky_temp, MPI.DOUBLE],
                                 recvbuf=[E_ky_long, (counts, None),
                                          MPI.DOUBLE])
            nkyE = self.nkyE
            E_ky = E_ky_long[0:nkyE]
            E_ky[1:nkyE] = E_ky[1:nkyE] + E_ky_long[self.nky_seq:nkyE:-1]
            E_ky = E_ky/self.deltaky

####        self.comm.barrier()
####        sleep(0.1)
####        print   self.rank,  'E_kx.sum() =', E_kx.sum()*self.deltakx, \
####                            'E_ky.sum() =', E_ky.sum()*self.deltaky,\
####                'diff = ', E_kx.sum()*self.deltakx-E_ky.sum()*self.deltaky
        return E_kx, E_ky

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def spectrum2D_from_fft(self,
                            np.ndarray[DTYPEf_t, ndim=2] E_fft):
        """Compute the 2D spectra. Return a dictionary."""
        cdef np.ndarray[DTYPEf_t, ndim=2] KK
        cdef np.uint32_t ikO, ik1, ikh, nkh
        cdef np.uint32_t nk0loc, nk1loc, rank, TRANSPOSED
        cdef DTYPEf_t E0D, kappa0D, deltakh, coef_share, energy
        cdef np.ndarray[DTYPEf_t, ndim=1] spectrum2D, khE

        KK = self.KK

        nk0loc = self.shapeK_loc[0]
        nk1loc = self.shapeK_loc[1]

        rank = self.rank

        if self.TRANSPOSED:
            TRANSPOSED = 1
        else:
            TRANSPOSED = 0

        deltakh = self.deltakh

        khE = self.khE
        nkh = self.nkhE

        spectrum2D = np.zeros([nkh])
        for ik0 in xrange(nk0loc):
            for ik1 in xrange(nk1loc):
                E0D = E_fft[ik0, ik1]/deltakh
                kappa0D = KK[ik0, ik1]

                if TRANSPOSED == 0:
                    if ik1 > 0:
                        E0D = E0D*2
                else:
                    if ik0 > 0 or rank > 0:
                        E0D = E0D*2

                ikh = int(kappa0D/deltakh)

                if ikh >= nkh-1:
                    ikh = nkh - 1
                    spectrum2D[ikh] += E0D
                else:
                    coef_share = (kappa0D - khE[ikh])/deltakh
                    spectrum2D[ikh] += (1-coef_share)*E0D
                    spectrum2D[ikh+1] += coef_share*E0D

        if nb_proc > 1:
            spectrum2D = comm.allreduce(spectrum2D, op=MPI.SUM)
        return spectrum2D

    def pdf_normalized(self, field, nb_bins=100):
        """Compute the normalized pdf"""

        field_max = field.max()
        field_min = field.min()
        # field_mean = field.mean()

        if nb_proc > 1:
            field_max = comm.allreduce(field_max, op=MPI.MAX)
            field_min = comm.allreduce(field_min, op=MPI.MIN)
            # field_mean = comm.allreduce(field_min, op=MPI.SUM)/nb_proc

        # rms = np.sqrt(np.mean( (field-field_mean)**2 ))
        # range_min = field_mean - 20*rms
        # range_max = field_mean + 20*rms

        # range_min = max(field_min, range_min)
        # range_max = min(field_max, range_max)

        range_min = field_min
        range_max = field_max

        if nb_proc == 1:
            pdf, bin_edges = np.histogram(field, bins=nb_bins,
                                          normed=True,
                                          range=(range_min, range_max))
        else:
            hist, bin_edges = np.histogram(field, bins=nb_bins,
                                           range=(range_min, range_max))
            hist = comm.allreduce(hist, op=MPI.SUM)
            pdf = hist/((bin_edges[1]-bin_edges[0])*hist.sum())
        return pdf, bin_edges

    def compute_increments_dim1(self,
                                np.ndarray[DTYPEf_t, ndim=2] var,
                                np.uint32_t irx):
        """Compute the increments of var over the dim 1."""
        cdef np.uint32_t iO, i1, n0, n1, n1new
        cdef np.ndarray[DTYPEf_t, ndim=2] inc_var
        n0 = var.shape[0]
        n1 = var.shape[1]
        n1new = n1 - irx
        inc_var = np.empty([n0, n1new])
        for i0 in xrange(n0):
            for i1 in xrange(n1new):
                inc_var[i0, i1] = (var[i0, i1+irx] - var[i0, i1])
        return inc_var

#### functions for initialisation of field
    def constant_arrayK(self, value=None, dtype=complex, SHAPE='LOC'):
        """Return a constant array in spectral space."""
        if SHAPE == 'LOC':
            shapeK = self.shapeK_loc
        elif SHAPE == 'SEQ':
            shapeK = self.shapeK_seq
        elif SHAPE == 'GAT':
            shapeK = self.shapeK_gat
        else:
            raise ValueError('SHAPE should be "LOC" or "SEQ"')
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
            raise ValueError('SHAPE should be "LOC" of "SEQ"')
        if value is None:
            field = np.empty(shapeX, dtype=dtype)
        elif value == 0:
            field = np.zeros(shapeX, dtype=dtype)
        else:
            field = value*np.ones(shapeX, dtype=dtype)
        return field

    def random_arrayK(self, SHAPE='LOC'):
        """Return a random array in spectral space."""
        if SHAPE == 'LOC':
            shapeK = self.shapeK_loc
        elif SHAPE == 'SEQ':
            shapeK = self.shapeK_seq
        elif SHAPE == 'GAT':
            shapeK = self.shapeK_gat
        else:
            raise ValueError('SHAPE should be "LOC", "GAT" or "SEQ"')
        a_fft = (np.random.random(shapeK)
                 + 1j*np.random.random(shapeK)
                 - 0.5 - 0.5j)
        return a_fft

    def random_arrayX(self, SHAPE='LOC'):
        """Return a random array in real space."""
        if SHAPE == 'LOC':
            shapeX = self.shapeX_loc
        elif SHAPE == 'SEQ':
            shapeX = self.shapeX_seq
        else:
            raise ValueError('SHAPE should be "LOC" or "SEQ"')
        return np.random.random(shapeX)

    def project_fft_on_realX_seq(
            self, np.ndarray[DTYPEc_t, ndim=2] f_fft):
        """Project the given field in spectral space such as its
        inverse fft is a real field."""
        cdef np.uint32_t nky_seq
        cdef np.uint32_t iky_ky0, iky_kyM, ikx_kx0, ikx_kxM,
        cdef np.uint32_t ikyp, ikyn
        cdef DTYPEc_t f_kp_kx0, f_kn_kx0, f_kp_kxM, f_knp_kxM

        nky_seq = self.shapeK_seq[0]

        iky_ky0 = 0
        iky_kyM = nky_seq/2
        ikx_kx0 = 0
        # ikx_kxM = self.nkx_seq-1
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

    def coarse_seq_from_fft_loc(self, f_fft, shapeK_loc_coarse):
        """Return a coarse field in K space."""
        nKyc = shapeK_loc_coarse[0]
        nKxc = shapeK_loc_coarse[1]

        if nb_proc > 1:
            fc_trans = np.empty([nKxc, nKyc], np.complex128)
            nKy = self.shapeK_seq[0]
            f1D_temp = np.empty([nKyc], np.complex128)

            for iKxc in xrange(nKxc):
                kx = self.deltakx*iKxc
                rank_iKx, iKxloc, iKyloc = self.where_is_wavenumber(kx, 0.)
                if rank == rank_iKx:
                    # create f1D_temp
                    for iKyc in xrange(nKyc):
                        if iKyc <= nKyc/2:
                            iKy = iKyc
                        else:
                            kynodim = iKyc - nKyc
                            iKy = kynodim + nKy
                        f1D_temp[iKyc] = f_fft[iKxloc, iKy]

                if rank_iKx != 0:
                    # message f1D_temp
                    if rank == 0:
                        # print 'f1D_temp', f1D_temp, f1D_temp.dtype
                        comm.Recv(
                            [f1D_temp, MPI.DOUBLE_COMPLEX],
                            source=rank_iKx, tag=iKxc)
                    elif rank == rank_iKx:
                        comm.Send(
                            [f1D_temp, MPI.DOUBLE_COMPLEX],
                            dest=0, tag=iKxc)
                if rank == 0:
                    # copy into fc_trans
                    fc_trans[iKxc] = f1D_temp.copy()
            fc_fft = fc_trans.transpose()

        else:
            nKy = self.shapeK_seq[0]
            nKx = self.shapeK_seq[1]
            fc_fft = np.empty([nKyc, nKxc], np.complex128)
            for iKyc in xrange(nKyc):
                if iKyc <= nKyc/2:
                    iKy = iKyc
                else:
                    kynodim = iKyc - nKyc
                    iKy = kynodim + nKy
                for iKxc in xrange(nKxc):
                    fc_fft[iKyc, iKxc] = f_fft[iKy, iKxc]
        return fc_fft

    def fft_loc_from_coarse_seq(self, fc_fft, shapeK_loc_coarse):
        """Return a large field in K space."""
        nKyc = shapeK_loc_coarse[0]
        nKxc = shapeK_loc_coarse[1]

        if nb_proc > 1:
            nKy = self.shapeK_seq[0]
            f_fft = self.constant_arrayK(value=0.)
            fc_trans = fc_fft.transpose()

            for iKxc in xrange(nKxc):
                kx = self.deltakx*iKxc
                rank_iKx, iKxloc, iKyloc = self.where_is_wavenumber(kx, 0.)
                fc1D = fc_trans[iKxc]
                if rank_iKx != 0:
                    # message fc1D
                    fc1D = np.ascontiguousarray(fc1D)
                    if rank == 0:
                        comm.Send(fc1D, dest=rank_iKx, tag=iKxc)
                    elif rank == rank_iKx:
                        comm.Recv(fc1D, source=0, tag=iKxc)
                if rank == rank_iKx:
                    # copy
                    for iKyc in xrange(nKyc):
                        if iKyc <= nKyc/2:
                            iKy = iKyc
                        else:
                            kynodim = iKyc - nKyc
                            iKy = kynodim + nKy
                        f_fft[iKxloc, iKy] = fc1D[iKyc]

        else:
            nKy = self.shapeK_seq[0]
            nKx = self.shapeK_seq[1]
            f_fft = np.zeros([nKy, nKx], np.complex128)
            for iKyc in xrange(nKyc):
                if iKyc <= nKyc/2:
                    iKy = iKyc
                else:
                    kynodim = iKyc - nKyc
                    iKy = kynodim + nKy
                for iKxc in xrange(nKxc):
                    f_fft[iKy, iKxc] = fc_fft[iKyc, iKxc]
        return f_fft

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def monge_ampere_from_fft(
            self, DTYPEc_t[:, :] a_fft, DTYPEc_t[:, :] b_fft):
        cdef Py_ssize_t i0, n0, i1, n1
        cdef DTYPEc_t[:, :] pxx_afft, pyy_afft, pxy_afft
        cdef DTYPEc_t[:, :] pxx_bfft, pyy_bfft, pxy_bfft
        cdef DTYPEf_t[:, :] mamp
        cdef DTYPEf_t[:, :] KX, KY, KX2, KY2
        cdef DTYPEf_t[:, :] pxx_a, pyy_a, pxy_a, pxx_b, pyy_b, pxy_b

        n0 = a_fft.shape[0]
        n1 = a_fft.shape[1]
        KX = self.KX
        KY = self.KY
        KX2 = self.KX2
        KY2 = self.KY2

        pxx_afft = np.empty([n0, n1], dtype=DTYPEc)
        pyy_afft = np.empty([n0, n1], dtype=DTYPEc)
        pxy_afft = np.empty([n0, n1], dtype=DTYPEc)
        pxx_bfft = np.empty([n0, n1], dtype=DTYPEc)
        pyy_bfft = np.empty([n0, n1], dtype=DTYPEc)
        pxy_bfft = np.empty([n0, n1], dtype=DTYPEc)

        for i0 in xrange(n0):
            for i1 in xrange(n1):
                pxx_afft[i0, i1] = - a_fft[i0, i1] * KX2[i0, i1]
                pyy_afft[i0, i1] = - a_fft[i0, i1] * KY2[i0, i1]
                pxy_afft[i0, i1] = - a_fft[i0, i1] * KX[i0, i1]*KY[i0, i1]
                pxx_bfft[i0, i1] = - b_fft[i0, i1] * KX2[i0, i1]
                pyy_bfft[i0, i1] = - b_fft[i0, i1] * KY2[i0, i1]
                pxy_bfft[i0, i1] = - b_fft[i0, i1] * KX[i0, i1]*KY[i0, i1]
        pxx_a = self.ifft2(pxx_afft)
        pyy_a = self.ifft2(pyy_afft)
        pxy_a = self.ifft2(pxy_afft)
        pxx_b = self.ifft2(pxx_bfft)
        pyy_b = self.ifft2(pyy_bfft)
        pxy_b = self.ifft2(pxy_bfft)

        mamp = np.empty_like(pxx_a)
        n0 = mamp.shape[0]
        n1 = mamp.shape[1]
        for i0 in xrange(n0):
            for i1 in xrange(n1):
                mamp[i0, i1] = (pxx_a[i0, i1] * pyy_b[i0, i1] +
                                pyy_a[i0, i1] * pxx_b[i0, i1] -
                                2 * pxy_a[i0, i1] * pxy_b[i0, i1])
        return np.array(mamp)

    def monge_ampere_from_fft_python(self, a_fft, b_fft):
        KX = self.KX
        KY = self.KY
        ifft2 = self.ifft2

        pxx_a = - ifft2(a_fft * KX**2)
        pyy_a = - ifft2(a_fft * KY**2)
        pxy_a = - ifft2(a_fft * KX * KY)

        pxx_b = - ifft2(b_fft * KX**2)
        pyy_b = - ifft2(b_fft * KY**2)
        pxy_b = - ifft2(b_fft * KX * KY)

        return pxx_a*pyy_b + pyy_a*pxx_b - 2*pxy_a*pxy_b

    def laplacian2_fft(self, DTYPEc_t[:, :] a_fft):
        cdef Py_ssize_t i0, n0, i1, n1
        cdef DTYPEc_t[:, :] lap2_afft = np.empty_like(a_fft)
        cdef DTYPEf_t[:, :] K4 = self.K4

        n0 = a_fft.shape[0]
        n1 = a_fft.shape[1]
        for i0 in xrange(n0):
            for i1 in xrange(n1):
                lap2_afft[i0, i1] = a_fft[i0, i1] * K4[i0, i1]
        return np.array(lap2_afft)

    def invlaplacian2_fft(self, DTYPEc_t[:, :] a_fft):
        cdef Py_ssize_t i0, n0, i1, n1
        cdef DTYPEc_t[:, :] invlap2_afft = np.empty_like(a_fft)
        cdef DTYPEf_t[:, :] K4_not0 = self.K4_not0

        n0 = a_fft.shape[0]
        n1 = a_fft.shape[1]

        for i0 in xrange(n0):
            for i1 in xrange(n1):
                invlap2_afft[i0, i1] = a_fft[i0, i1] / K4_not0[i0, i1]

        if rank == 0:
            invlap2_afft[0, 0] = 0.
        return np.array(invlap2_afft)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray dealiasing_variable(np.ndarray[DTYPEc_t, ndim=2] ff_fft,
                                    np.ndarray[DTYPEb_t, ndim=2] where,
                                    int nK0loc, int nK1loc):
    cdef np.uint32_t iKO, iK1
    for iK0 in range(nK0loc):
        for iK1 in range(nK1loc):
            if where[iK0, iK1]:
                ff_fft[iK0, iK1] = 0.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray dealiasing_setofvar(np.ndarray[DTYPEc_t, ndim=3] setofvar_fft,
                                    np.ndarray[DTYPEb_t, ndim=2] where,
                                    Py_ssize_t n0, Py_ssize_t n1):
    cdef Py_ssize_t ik, nk, i0, i1
    nk = setofvar_fft.shape[0]

    for i0 in xrange(n0):
        for i1 in xrange(n1):
            if where[i0, i1]:
                for ik in xrange(nk):
                    setofvar_fft[ik, i0, i1] = 0.
