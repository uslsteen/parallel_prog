#include <iostream>
#include <cmath>

#include <mpich/mpi.h>

const double START = 0;
const double END = 2;
const double STEP = 1 / 1000;

inline double fn(double x) {
    return std::sqrt(4 - x * x); 
}

void usage(char* path) {
    std::cout << "USAGE: mpirun -np <nproc> "  << path << " <num>\n" << std::endl;
}

int main(int argc, char** argv) {
    int rank = 0, commsize = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);

    if (argc == 2) {
        
    }
    else
        usage(argv[0]);

    return 0;
}