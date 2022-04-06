#include <mpi.h>
#include <stdio.h>

void dump(int data, int rank) {
    printf("data=%d, rank=%d\n", data, rank);
}

void send_last(int data, int commsize) {
    
    dump(data, 0);
    int tag = 0, dest = 1, src = commsize - 1;

    MPI_Send(&data, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
    MPI_Recv(&data, 1, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    dump(++data, 0);
}

void ring_simulator(int commsize, int rank) {
    
    int src_rank = rank - 1, 
        dest_rank = rank == commsize - 1 ? 0 : rank + 1;
    int data = 0, tag = 0;

    if (rank == 0) {
        send_last(data, commsize);
        return;
    }
    
    MPI_Recv(&data, 1, MPI_INT, src_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    dump(++data, rank);
    MPI_Send(&data, 1, MPI_INT, dest_rank, tag, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    int commsize, my_rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    ring_simulator(commsize, my_rank);

    MPI_Finalize();
}