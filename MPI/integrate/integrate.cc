#include <array>
#include <iostream>
#include <cmath>
#include <vector>

// #include <mpich/mpi.h>
#include <mpi.h>


const double START = 0;
const double END = 2;
const double STEP = double(1)/1000;
const double LEN = END - START;

inline double fn(double x) {
    return std::sqrt(4 - x * x); 
}

void usage(char* path) {
    std::cout << "USAGE: mpirun -np <nproc> "  << path << std::endl;
}

double integrate_root(int commsize);
void integrate();

double calc(std::array<int, 2>& cur_task) {

    double start = START + cur_task[0] * LEN * STEP,
           end = START + (cur_task[0] + cur_task[1]) * LEN * STEP,
           step = (end - start) / cur_task[1];

    double sum = 0, part_integr = 0;

    for (double x_i = start + step; x_i < end; x_i += step)
        sum += fn(x_i);

    part_integr = step * ((fn(start) + fn(end))/2 + sum);

    return part_integr;
}

int main(int argc, char** argv) {
    int rank = 0, commsize = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);

    if (argc == 1) {
        
        if (rank == 0) {
            double start = MPI_Wtime();
            std::cout << "Result=" << integrate_root(commsize) << std::endl;
            double end = MPI_Wtime();

            std::cout << "time=" << (end - start) * 1000 << "msecs" << std::endl;            
        }
        else 
            integrate();
    }
    else
        usage(argv[0]);

    return 0;
}

void integrate() {
    std::array<int, 2> cur_task{};

    uint32_t tag = 0, root = 0;
    MPI_Recv((int*)&cur_task, 2, MPI_INT, root, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    double part_integr = calc(cur_task);
    MPI_Send(&part_integr, 1, MPI_DOUBLE, root, tag, MPI_COMM_WORLD);
}

double integrate_root(int commsize) {
    uint32_t tag = 0, nsteps = 1 / STEP;

    int cur_job = nsteps / commsize,
        part_job = nsteps % commsize;

    std::vector<std::array<int, 2>> tasks(commsize);

    for (size_t i = 0; i < part_job; ++i) {
        tasks[i][0] = (cur_job + 1) * i;
        tasks[i][1] = (cur_job + 1);
    }

    for (size_t i = part_job; i < commsize; ++i) {
        tasks[i][0] = cur_job * i + part_job;
        tasks[i][1] = cur_job;
    }

    for (size_t i = 1; i < commsize; ++i) 
        MPI_Send((int*)&tasks[i], 2, MPI_INT, i, tag, MPI_COMM_WORLD);

    double res = calc(tasks[0]), part_integr = 0;

    for (size_t i = 1; i < commsize; ++i) {
        MPI_Recv(&part_integr, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        res += part_integr;
    }

    return res; 
}