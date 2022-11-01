#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

void sum_send(int32_t num, int32_t cur_rank, int32_t commsize) {
    
    int32_t tag = 0, dest = 0;
    double cur_job = num / commsize;
    double data = 0;

    uint32_t cur_start = 1 + cur_job * (cur_rank - 1);

    //! NOTE: if cur_rank is last rank in communicator
    if (cur_rank == commsize)
        cur_job += num % commsize;
        
    for (uint32_t i = cur_start, cur_end = cur_start + cur_job; i < cur_end; ++i)
        data += (double) 1 / i;

    MPI_Send(&data, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
}

double sum_recv(uint32_t commsize) {

    int32_t tag = 0;
    double res_sum = 0;

    for (uint32_t i = 1; i < commsize; ++i) {
        double data = 0;
        MPI_Recv(&data, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
        res_sum += data;
    }

    return res_sum;
}

void usage(char* exec_path) {    
    printf("USAGE: mpirun -np <nproc> %s <num>\n", exec_path);
}

int main(int argc, char** argv) {
    int32_t commsize, rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc == 2) {
        double num = atoi(argv[1]);

        if (rank == 0)
            printf("%lf", sum_recv(commsize));
        else
            sum_send(num, rank, commsize - 1);
    }
    else
        usage(argv[0]);

    MPI_Finalize();
}
