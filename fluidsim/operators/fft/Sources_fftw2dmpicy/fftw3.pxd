



cdef extern from "complex.h":
    pass

cdef extern from "stddef.h":
    ctypedef void ptrdiff_t

cdef extern from "fftw3.h":
    ctypedef struct fftw_plan_s:
        pass
    ctypedef fftw_plan_s *fftw_plan
    ctypedef struct fftw_iodim:
        int n
        int ins "is"
        int ous "os"

    fftw_plan fftw_plan_dft_r2c_2d(int n0,
                                   int n1,
                                   double* in_,
                                   complex* out_,
                                   unsigned flags)

    fftw_plan fftw_plan_dft_c2r_2d(int n0,
                                   int n1,
                                   complex* in_,
                                   double* out_,
                                   unsigned flags)

    double* fftw_alloc_real(size_t n)
    complex* fftw_alloc_complex(size_t n)
    void fftw_execute(fftw_plan plan)
    void fftw_destroy_plan(fftw_plan plan)
    void fftw_free(void *mem)


cdef enum:
    FFTW_FORWARD = -1
    FFTW_BACKWARD = +1
    FFTW_MEASURE = 0
    FFTW_DESTROY_INPUT =  (1 << 0)
    FFTW_UNALIGNED = (1 << 1)
    FFTW_CONSERVE_MEMORY = (1 << 2)
    FFTW_EXHAUSTIVE = (1 << 3) # /* NO_EXHAUSTIVE is default */
    FFTW_PRESERVE_INPUT = (1 << 4) # /* cancels FFTW_DESTROY_INPUT */
    FFTW_PATIENT = (1 << 5) # /* IMPATIENT is default */
    FFTW_ESTIMATE = (1 << 6)
    FFTW_MPI_TRANSPOSED_IN = (1U << 29)
    FFTW_MPI_TRANSPOSED_OUT = (1U << 30)
    FFTW_MPI_DEFAULT_BLOCK = 0


