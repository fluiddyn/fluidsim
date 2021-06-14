
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

using std::cout;
using std::endl;

int main(int argc, char** argv) {

    double t_start, duration;
    int version, sub_version;

    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (world_rank == 0) {
        char lib_version[MPI_MAX_LIBRARY_VERSION_STRING];
        int resultlen;
        MPI_Get_library_version(lib_version, &resultlen);
        cout << lib_version << endl;
        MPI_Get_version(&version, &sub_version);
        printf("MPI Version %d.%d\n", version, sub_version);
    }

    int size = 51200;
    double* numbers;

    for (int i = 0; i < 11; i++) {
        size = 2 * size;

        numbers = (double*) malloc(size * sizeof(double));

        if (world_rank == 0) {
            MPI_Send(numbers, size, MPI_DOUBLE, 1, 77, MPI_COMM_WORLD);
            t_start = MPI_Wtime();
            MPI_Send(numbers, size, MPI_DOUBLE, 1, 77, MPI_COMM_WORLD);
            duration = MPI_Wtime() - t_start;
            printf("%.3e s for %10d floats (%.3f Gb/s)\n", duration, size, 64e-9 * size / duration);
        } else if (world_rank == 1) {
            MPI_Recv(numbers, size, MPI_DOUBLE, 0, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(numbers, size, MPI_DOUBLE, 0, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        free(numbers);
    }

    MPI_Finalize();
}