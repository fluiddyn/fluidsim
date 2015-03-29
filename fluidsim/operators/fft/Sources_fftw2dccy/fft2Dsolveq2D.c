/* test_fftw3_2Dmpi_simple program 

export USR_PERSO='/home/pierre/usr'
export USR_PERSO='/scratch/augier/usr'

Compiled with:
gcc -O3 fft2Dsolveq2D.c -I$USR_PERSO/include/ \
$USR_PERSO/lib/libfftw3.so -lm -o fft2Dsolveq2D

Create the library:
gcc -c -fPIC fft2Dsolveq2D.c -I$USR_PERSO/include -o fft2Dsolveq2D.o
gcc fft2Dsolveq2D.o -shared -o libfft2Dsolveq2D.so $USR_PERSO/lib/libfftw3.so -lm
mv libfft2Dsolveq2D.so $USR_PERSO/lib



And on ferlin:
gcc -O3 fft2Dsolveq2D.c -I${FFTW_HOME}/double/include \
-L${FFTW_HOME}/double/lib -lfftw3 -lm -o fft2Dsolveq2D

Create the library:
gcc -c -fPIC fft2Dsolveq2D.c -I${FFTW_HOME}/double/include -o fft2Dsolveq2D.o

gcc -shared -o libfft2Dsolveq2D.so fft2Dsolveq2D.o \
${FFTW_HOME}/double/lib/libfftw3.so -lm 


mpicc -c -fPIC fft2Dmpisolveq2D.c -I${FFTW_HOME}/double/include \
-o fft2Dmpisolveq2D.o
mpicc -shared -o libfft2Dmpisolveq2D.so fft2Dmpisolveq2D.o \
-L${FFTW_HOME}/double/lib -lm -lfftw3




execute with
mpirun -np 2 ./fft2Dsolveq2D
(the 2 after "-np" is the number of processors)
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include <sys/time.h>


#include <unistd.h>

#include <math.h>

#include "fft2Dsolveq2D.h"




Util_fft init_Util_fft(int N0, int N1)
    {
    Util_fft uf;
    struct timeval start_time, end_time;
    double total_usecs;
    int ii, jj, irank, iX0;
    int istride = 1, ostride = 1;
    int howmany, sign;


/*    printf("init_util_fft, N0 = %d, N1 = %d\n", N0, N1);*/

    uf.N0 = N0;
    uf.N1 = N1;

    /* y corresponds to dim 0 in physical space */
    /* x corresponds to dim 1 in physical space */
    uf.ny = N0;
    uf.nx = N1;

    uf.nX0 = N0;
    uf.nX1 = N1;
    uf.nX0loc = N0;
    uf.nXyloc = uf.nX0loc;

    uf.nKx = uf.nx/2;
    uf.nKxloc = uf.nKx;
    uf.nKy = uf.ny;

    /* This 2D fft is transposed */
    uf.nK0 = N1/2;
    uf.nK0loc = uf.nK0;
    uf.nK1 = N0;

    uf.coef_norm = N0*N1;

    uf.flags = FFTW_MEASURE;
/*    flags = FFTW_ESTIMATE;*/
/*    uf.flags = FFTW_PATIENT;*/

    uf.arrayX    = (double*) fftw_malloc(sizeof(double)*uf.nX0loc*N1);
    uf.arrayK_pR = (fftw_complex*) fftw_malloc( sizeof(fftw_complex)
                                                *uf.nX0loc*(uf.nKx+1));
    uf.arrayK_pC = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*uf.nKxloc*N0);


/*    if ((uf.rank)==0) printf("create plans\n");*/
    gettimeofday(&start_time, NULL);
/*    plan = fftw_plan_many_dft(int rank, const int *n, int howmany,*/
/*                                  fftw_complex *in, const int *inembed,*/
/*                                  int istride, int idist,*/
/*                                  fftw_complex *out, const int *onembed,*/
/*                                  int ostride, int odist,*/
/*                                  int sign, unsigned flags);*/

    howmany = uf.nX0loc;
    uf.plan_r2c = fftw_plan_many_dft_r2c(   1, &N1, howmany,
                                            uf.arrayX, NULL,
                                            istride, N1,
                                            uf.arrayK_pR, NULL,
                                            ostride, uf.nKx+1,
                                            uf.flags);

    uf.plan_c2r = fftw_plan_many_dft_c2r(   1, &N1, howmany,
                                            uf.arrayK_pR, NULL,
                                            istride, uf.nKx+1,
                                            uf.arrayX, NULL,
                                            ostride, N1,
                                            uf.flags);

    howmany = uf.nKxloc;
    sign = FFTW_FORWARD;
    uf.plan_c2c_fwd = fftw_plan_many_dft(   1, &N0, howmany,
                                            uf.arrayK_pC, &N0,
                                            istride, N0,
                                            uf.arrayK_pC, &N0,
                                            ostride, N0,
                                            sign, uf.flags);
    sign = FFTW_BACKWARD;
    uf.plan_c2c_bwd = fftw_plan_many_dft(   1, &N0, howmany,
                                            uf.arrayK_pC, &N0,
                                            istride, N0,
                                            uf.arrayK_pC, &N0,
                                            ostride, N0,
                                            sign, uf.flags);

    gettimeofday(&end_time, NULL);
    total_usecs = (end_time.tv_sec-start_time.tv_sec) + 
        (end_time.tv_usec-start_time.tv_usec)/1000000.;
/*    printf ("               done in %f s\n", */
/*            total_usecs);*/

    for (iX0=0;iX0<uf.nX0loc;iX0++)
        uf.arrayK_pR[iX0*(uf.nKx+1)+uf.nKx] = 0.;



/*    for (ii = 0; ii < uf.nX0loc; ++ii) for (jj = 0; jj < uf.nKx+1; ++jj)*/
/*    {*/
/*    printf("%i , uf.arrayK_pR[%i*(uf.nKx+1) + %i] = (%6.4f, %6.4f)\n", */
/*              uf.rank, ii, jj,*/
/*             creal(uf.arrayK_pR[ii*(uf.nKx+1) + jj]), */
/*               cimag(uf.arrayK_pR[ii*(uf.nKx+1) + jj]));*/
/*                }*/




    return uf;
}



void destroy_Util_fft(Util_fft uf)
    {
/*    if ((uf.rank)==0) printf("destroy_util_fft\n");*/
    fftw_destroy_plan(uf.plan_r2c);
    fftw_destroy_plan(uf.plan_c2c_fwd);
    fftw_destroy_plan(uf.plan_c2c_bwd);
    fftw_destroy_plan(uf.plan_c2r);
    fftw_free(uf.arrayX);
    fftw_free(uf.arrayK_pR);
    fftw_free(uf.arrayK_pC);
}









void fft2D(Util_fft uf, double *fieldX, fftw_complex *fieldK)
    {
    int ii, jj;
    /*use memcpy(void * destination, void * source, size_t bytes); */

    memcpy(uf.arrayX, fieldX, uf.nX0loc*uf.nX1*sizeof(double));


/*       for (ii = 0; ii < uf.nX0loc; ++ii) for (jj = 0; jj < uf.N1; ++jj)*/
/*            {*/
/*            printf( "%d , uf.arrayX[%d,%d] = %6.4f\n", */
/*                    uf.rank, ii, jj, uf.arrayX[ii*uf.N1+jj]);*/
/*            }*/


    fftw_execute(uf.plan_r2c);

/*     printf("print uf.arrayK_pR after alltoall\n");*/

/*     for (ii = 0; ii < uf.nX0loc; ++ii) for (jj = 0; jj < uf.nKx+1; ++jj)*/
/*       {*/
/*       printf("%i , uf.arrayK_pR[%i*(uf.nKx+1) + %i] = (%6.4f, %6.4f)\n", */
/*                        uf.rank, ii, jj,*/
/*                        creal(uf.arrayK_pR[ii*(uf.nKx+1)+jj]), */
/*                        cimag(uf.arrayK_pR[ii*(uf.nKx+1)+jj]));*/
/*                }*/

/*    second step: transpose...*/
    for (ii = 0; ii < uf.nX0; ++ii) for (jj = 0; jj < uf.nKx; ++jj)
        uf.arrayK_pC[jj*uf.nX0+ii] = 
          uf.arrayK_pR[ii*(uf.nKx+1)+jj];


/*       for (ii = 0; ii < uf.nKxloc; ++ii) for (jj = 0; jj < uf.N0; ++jj)*/
/*           {*/
/*        printf("%i , uf.arrayK_pC[%i*uf.N0 + %i] = (%6.4f, %6.4f)\n", */
/*                        uf.rank, ii, jj,*/
/*                        creal(uf.arrayK_pC[ii*uf.N0 + jj]), */
/*                        cimag(uf.arrayK_pC[ii*uf.N0 + jj]));*/
/*                }*/



    fftw_execute(uf.plan_c2c_fwd);


/*      for (ii = 0; ii < uf.nKxloc; ++ii) for (jj = 0; jj < uf.N0; ++jj)*/
/*           {*/
/*         printf("%i , uf.arrayK_pC[%i*uf.N0 + %i] = (%6.4f, %6.4f)\n", */
/*                        uf.rank, ii, jj,*/
/*                        creal(uf.arrayK_pC[ii*uf.N0 + jj]), */
/*                        cimag(uf.arrayK_pC[ii*uf.N0 + jj]));*/
/*                }*/



    for (ii=0; ii<uf.nKxloc*uf.nKy; ii++)
        fieldK[ii]  = uf.arrayK_pC[ii]/uf.coef_norm;

    }




void ifft2D(Util_fft uf, fftw_complex *fieldK, double *fieldX)
    {
    int ii, jj, irank;
    /*use memcpy(void * destination, void * source, size_t bytes); */
    memcpy(uf.arrayK_pC, fieldK, uf.nKxloc*uf.nKy*sizeof(fftw_complex));
    fftw_execute(uf.plan_c2c_bwd);

    /* second step: transpose...*/
    for (ii = 0; ii < uf.nKx; ++ii) for (jj = 0; jj < uf.nX0; ++jj)
          uf.arrayK_pR[jj*(uf.nKx+1)+ii] =
            uf.arrayK_pC[ii*uf.nX0+jj];

    /* these modes (nx/2+1=N1/2+1) have to be settled to zero*/
    for (ii = 0; ii < uf.nX0loc; ++ii) 
        uf.arrayK_pR[ii*(uf.nKx+1) + uf.nKx] = 0.;

    fftw_execute(uf.plan_c2r);
    memcpy(fieldX,uf.arrayX, uf.nX0loc*uf.nX1*sizeof(double));
    }


void time_execute(Util_fft uf, double *fieldX, 
                  fftw_complex *fieldK, int nb_time_execute)
    {
    int ii;
    struct timeval start_time, end_time;
    double total_usecs;

    if (uf.rank==0) printf("timer...\n");
    gettimeofday(&start_time, NULL);
    for (ii=0; ii<nb_time_execute; ii++)
    {
        fft2D(uf, fieldX, fieldK);
        ifft2D(uf, fieldK, fieldX);
    }
    gettimeofday(&end_time, NULL);


    total_usecs = (end_time.tv_sec-start_time.tv_sec) + 
        (end_time.tv_usec-start_time.tv_usec)/1000000.;
    printf ("%d times forward and backward sequencial FFT: %f s\n", 
            nb_time_execute ,total_usecs); 
}



int main(int argc, char **argv)
    {
    const int N0 = 4, N1 = 4;
    Util_fft uf;
    double * fieldX, * fieldX_0;
    fftw_complex * fieldK, * fieldK_0;
    int ii, jj, irank;
    double energyK, energyX, energy2;
    int coef;


    uf = init_Util_fft(N0,N1);

    srand(time(NULL)+uf.rank*uf.nb_proc);

    printf( "I'm rank (processor number) %i of size %i\n", 
            uf.rank, uf.nb_proc);

    fieldX = (double *) malloc(uf.nX0loc*uf.nX1 * sizeof(double));
    fieldX_0 = (double *) malloc(uf.nX0loc*uf.nX1 * sizeof(double));
    fieldK = (fftw_complex *) malloc(uf.nKxloc*uf.nKy * sizeof(fftw_complex));
    fieldK_0 = (fftw_complex *) malloc(uf.nKxloc*uf.nKy * sizeof(fftw_complex));


/*    time_execute(uf, fieldX, fieldK, 100);*/



    for (ii = 0; ii < uf.nX0loc; ++ii) for (jj = 0; jj < uf.nX1; ++jj)
        {
/*        fieldX[ii*uf.nX1+jj] = 2.;*/
        fieldX[ii*uf.nX1+jj] = rand()/(double)RAND_MAX -0.5;
        }

    for (ii = 0; ii < uf.nKxloc; ++ii) for (jj = 0; jj < uf.nKy; ++jj)
        {
        fieldK[ii*uf.nKy+jj] = 0.;
        }

/* We have to project on the space available for this library */
    fft2D(uf, fieldX, fieldK);
    ifft2D(uf, fieldK, fieldX);



    for (ii = 0; ii < uf.nKxloc; ++ii) for (jj = 0; jj < uf.nKy; ++jj)
        fieldK_0[ii*uf.nKy+jj] = fieldK[ii*uf.nKy+jj];

    for (ii = 0; ii < uf.nX0loc; ++ii) for (jj = 0; jj < uf.nX1; ++jj)
        fieldX_0[ii*uf.nX1+jj] = fieldX[ii*uf.nX1+jj];



    /* if (uf.rank==0)  */
    /* { */
/*    fieldK[0*uf.nKy+0] = 1.;*/

/*    fieldK[0*uf.nKy+1] = 1.I;*/
/*    fieldK[0*uf.nKy+3] = -1.I;*/

    /* fieldK[0*uf.nKy+2] = 1.; */
    /* } */


/*    if (uf.rank==1) */
/*    {*/
/*    fieldK[0*uf.nKy+0] = 1.;*/

/*    fieldK[0*uf.nKy+1] = 1.I;*/
/*    fieldK[0*uf.nKy+3] = -1.I;*/

/*    fieldK[0*uf.nKy+2] = 1.I;*/
/*    }*/








    time_execute(uf, fieldX, fieldK, 100);



    for (ii = 0; ii < uf.nKxloc; ++ii) for (jj = 0; jj < uf.nKy; ++jj)
        {
        printf("%i , fieldK[%i*nKxloc + %i] = (%6.4f, %6.4f)\n", 
               uf.rank, ii, jj,
               creal(fieldK[ii*uf.nKy + jj]),  
               cimag(fieldK[ii*uf.nKy + jj]));
        }


    for (ii = 0; ii < uf.nX0loc; ++ii) for (jj = 0; jj < uf.nX1; ++jj)
        {
        printf( "%d , fieldX[%d,%d] = %+6.4f\n", 
                uf.rank, ii, jj, fieldX[ii*uf.nX1+jj]);
        }

    for (ii = 0; ii < uf.nX0loc; ++ii) for (jj = 0; jj < uf.nX1; ++jj)
        {
         printf("%d , (fieldX - fieldX_0)[%d,%d] = %+6.4f\n", 
                 uf.rank, ii, jj, 
                fieldX[ii*uf.nX1+jj]-fieldX_0[ii*uf.nX1+jj]);
        }

    energyX = 0.;
    for (ii = 0; ii < uf.nX0loc; ++ii) for (jj = 0; jj < uf.nX1; ++jj)
        {
        energyX += fieldX[ii*uf.nX1+jj]*fieldX[ii*uf.nX1+jj];
        }
    energyX = energyX/uf.coef_norm;

    printf( "energyX = %6.4f\n", energyX);


    energyK = 0.;
    for (ii = 0; ii < uf.nKxloc; ++ii) for (jj = 0; jj < uf.nKy; ++jj)
        {
        if (ii==0)
            coef = 1;
        else
            coef = 2;
        energyK += pow(cabs(fieldK[ii*uf.nKy + jj]), 2) *coef;
        }

     printf( "energyK = %6.4f\n", energyK);






    free(fieldX);
    free(fieldX_0);
    free(fieldK);

    destroy_Util_fft(uf);
    }



