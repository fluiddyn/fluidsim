

typedef struct 
{
  /* X and K denote physical and Fourier spaces. */
  /* y corresponds to dim 0 in physical space */
  /* x corresponds to dim 1 in physical space */
  int N0, N1, nX0, nX1, nX0loc;
  int ny, nx, nXyloc;
  /* y corresponds to dim 1 in Fourier space */
  /* x corresponds to dim 0 in Fourier space */
  int nK0, nK1, nK0loc; 
  int nKx, nKy, nKxloc;
  int coef_norm;
  fftw_plan plan_r2c, plan_c2c_fwd, plan_c2r, plan_c2c_bwd;
  double *arrayX;
  fftw_complex *arrayK_pR, *arrayK_pC;
  unsigned flags;
  int rank, nb_proc, irank;
} Util_fft;

Util_fft init_Util_fft(int N0, int N1);
void destroy_Util_fft(Util_fft uf);
void fft2D(Util_fft uf, double *fieldX, fftw_complex *fieldK);
void ifft2D(Util_fft uf, fftw_complex *fieldK, double *fieldX);

