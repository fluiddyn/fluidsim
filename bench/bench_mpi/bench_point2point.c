
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <time.h>


int main(int argc, char** argv) {

    struct timespec tstart={0,0}, tend={0,0};

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int size = 100;
    double* numbers;

    for (int i = 0; i < 16; i++) {
        size = 2 * size;

        numbers = (double*) malloc(size * sizeof(double));

        if (world_rank == 0) {
            MPI_Send(numbers, size, MPI_DOUBLE, 1, 77, MPI_COMM_WORLD);
        } else if (world_rank == 1) {
            MPI_Recv(numbers, size, MPI_DOUBLE, 0, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tstart);

        if (world_rank == 0) {
            MPI_Send(numbers, size, MPI_DOUBLE, 1, 77, MPI_COMM_WORLD);
        } else if (world_rank == 1) {
            MPI_Recv(numbers, size, MPI_DOUBLE, 0, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == 0) {
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tend);
            double duration = (double)tend.tv_sec + 1.0e-9*tend.tv_nsec - (double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec;
            printf("%.3e s for %8d floats (%.3f Gb/s)\n", duration, size, 64e-9 * size / duration);
        }

        free(numbers);
    }


    MPI_Finalize();
}