


cdef extern from "fftw3.h":
    ctypedef struct fftw_plan_s:
        pass
    ctypedef fftw_plan_s *fftw_plan


from mpi4py cimport MPI

cdef extern from "fftw3-mpi.h":
    size_t fftw_mpi_local_size_2d(size_t n0, size_t n1, MPI.MPI_Comm comm,
                                  ptrdiff_t *local_n0, ptrdiff_t *local_0_start)

    size_t fftw_mpi_local_size_2d_transposed(
        size_t n0, size_t n1, MPI.MPI_Comm comm,
        ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
        ptrdiff_t *local_n1, ptrdiff_t *local_1_start)

    fftw_plan fftw_mpi_plan_dft_r2c_2d(int n0,
                                       int n1,
                                       double *in_,
                                       complex *out,
                                       MPI.MPI_Comm comm,
                                       unsigned flags)
    fftw_plan fftw_mpi_plan_dft_c2r_2d(int n0,
                                       int n1,
                                       complex *in_,
                                       double *out,
                                       MPI.MPI_Comm comm,
                                       unsigned flags)

    void fftw_mpi_init()
    void fftw_mpi_cleanup()
