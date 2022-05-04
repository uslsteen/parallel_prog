#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int SUM_TAG = 1;
int FACT_TAG = 1;

void exp_send(int32_t num, int32_t cur_rank, int32_t commsize) {
    
    int32_t dest = 0;
    double cur_job = num / commsize;
    double sum_part = 0, fact_part = 1;

    uint32_t cur_start = 1 + cur_job * (cur_rank - 1);

    //! NOTE: if cur_rank is last rank in communicator
    if (cur_rank == commsize)
        cur_job += num % commsize;
    
    uint32_t cur_end = cur_start + cur_job;
        
    for (uint32_t i = cur_start; i < cur_end; ++i) {
        fact_part /= i;
        sum_part += fact_part;        
    }

    MPI_Send(&sum_part, 1, MPI_DOUBLE, dest, SUM_TAG, MPI_COMM_WORLD);           
    MPI_Send(&fact_part, 1, MPI_DOUBLE, dest, FACT_TAG, MPI_COMM_WORLD);
}

double exp_recv(uint32_t commsize) {

    double exp_res = 1, cur_fact = 1;

    for (uint32_t i = 1; i < commsize; ++i) {
        double data = 0, fact_mult = 0;

        MPI_Recv(&data, 1, MPI_DOUBLE, i, SUM_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
        MPI_Recv(&fact_mult, 1, MPI_DOUBLE, i, FACT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
    
        exp_res += data * cur_fact;
        cur_fact *= fact_mult; 
    }

    return exp_res;
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

        if (rank == 0) {
            double start = MPI_Wtime();
            printf("Calc exp=%lf\n", exp_recv(commsize));
            double end = MPI_Wtime();

            printf("Exec time=%lf\n", end - start);
        }
        else
            exp_send(num, rank, commsize - 1);
    }
    else
        usage(argv[0]);

    MPI_Finalize();
}