
from mpi4py cimport MPI

cdef extern from "complex.h":
    pass

cdef extern from "fftw3.h":
    ctypedef struct fftw_plan_s:
        pass
    ctypedef fftw_plan_s *fftw_plan

cdef extern from "libcfftw2dmpi.h":
    ctypedef struct Util_fft:
        # X and K denote physical and Fourier spaces
        # y corresponds to dim 0 in physical space
        # x corresponds to dim 1 in physical space
        int N0, N1, nX0, nX1, nX0loc
        int ny, nx, nXyloc
        # y corresponds to dim 1 in Fourier space
        # x corresponds to dim 0 in Fourier space
        int nK0, nK1, nK0loc
        int nKx, nKy, nKxloc
        int coef_norm
        fftw_plan plan_r2c, plan_c2c_fwd, plan_c2r, plan_c2c_bwd
        double *arrayX
        complex *arrayK_pR
        complex *arrayK_pC
        unsigned flags
        int rank, nb_proc, irank
        MPI.MPI_Datatype MPI_type_column, MPI_type_block

    Util_fft init_Util_fft(int N0, int N1)
    void destroy_Util_fft(Util_fft uf)
    void fft2D(Util_fft uf, double *fieldX, complex *fieldK)
    void ifft2D(Util_fft uf, complex *fieldK, double *fieldX)
