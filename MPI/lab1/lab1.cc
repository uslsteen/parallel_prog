#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

#include <mpich/mpi.h>


const double PI = std::atan(1) * 4;
constexpr double T = 1,
                 X = 1;

//! NOTE: step along the x and t axes
constexpr double h = 1e-2,
                 tau = 1e-2;

//! NOTE: num of steps
//! NOTE: check correctness of rounding
constexpr uint32_t x_steps = static_cast<uint32_t>(X / h),
                   t_steps = static_cast<uint32_t>(T / tau);

inline double f(double x, double t) {
    return x + t;
}

inline double phi(double x) { return std::cos(PI * x); };
inline double psi(double t) { return std::exp(-t); };

std::ostream& operator <<(std::ostream &os, const std::vector<int>& vec) {
     for (auto && it : vec)
        os << it << std::endl;

    return os;
}

std::vector<std::vector<double>> init();

void dump(std::vector<double>& data, std::string&& out_f);

void do_linear(std::vector<std::vector<double>>& data);

void do_parallel(std::vector<std::vector<double>>& data, int32_t rank, int32_t commsize);
void do_main_job(std::vector<std::vector<double>>& data, int32_t rank, int32_t commsize);
std::vector<double> do_last_part(std::vector<std::vector<double>>& data, int32_t rank, int32_t commsize);

int main(int argc, char** argv) {
    auto&& data = init();

    int rank = 0, commsize = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    
    if (commsize == 1)
        do_linear(data);
    else
        do_parallel(data, rank, commsize);

    MPI_Finalize();
    return 0;
}

void do_parallel(std::vector<std::vector<double>>& data, int32_t rank, int32_t commsize) {

    double start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    do_main_job(data, rank, commsize);
    std::vector<double> result = do_last_part(data, rank, commsize);
    result.resize(x_steps * t_steps);

    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = MPI_Wtime();

    if (rank == 0) {
        std::cout << end - start << " " << commsize << std::endl;
        dump(result, std::string{"data_parallel.out"});
    }
}

void do_main_job(std::vector<std::vector<double>>& data, int32_t rank, int32_t commsize) {
    for (int32_t t = rank; t < t_steps; t += commsize) {
        for (int32_t x = 1; x < x_steps; ++x) {
            if (t != 0) {
                int src = rank ? rank - 1 : commsize - 1;
                MPI_Recv(&(data[t - 1][x]), 1, MPI_DOUBLE, src, x, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                data[t][x] =  data[t-1][x] - (tau / h) * (data[t-1][x] - data[t-1][x - 1]) + tau * f(x * h, (t - 1) * tau);    
            }
            int dest = (t + 1) % commsize;
            MPI_Send(&(data[t][x]), 1, MPI_DOUBLE, dest, x, MPI_COMM_WORLD);
        }
    }
}

std::vector<double> do_last_part(std::vector<std::vector<double>>& data, int32_t rank, int32_t commsize) {
    std::vector<double> result{};
    int32_t res_t_steps = t_steps + commsize - t_steps % commsize;
    
    if (rank == 0) 
        result.resize(x_steps * res_t_steps);

    std::vector<double> zeros(x_steps, 0);

    for (int32_t t = rank; t < res_t_steps; t += commsize) {
        if (commsize > 1) {
            auto *src_buf = t < t_steps ? data[t].data() : zeros.data();
            MPI_Gather(src_buf, x_steps, MPI_DOUBLE,
                       result.data() + t * x_steps, x_steps, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        else if (t < t_steps)
            std::copy(data[t].begin(), data[T].end(), result.begin() + t * x_steps);
        else
            break;
    }
    return result;
}

void do_linear(std::vector<std::vector<double>>& data) {
    double start = MPI_Wtime();
    std::vector<double> result{};
    
    for (uint32_t t = 0; t < t_steps - 1; ++t)
        for (uint32_t x = 1; x < x_steps; ++x)
            data[t + 1][x] =  data[t][x] - (tau / h) * (data[t][x] - data[t][x - 1]) + tau * f(x * h, t * tau);

    for (int32_t t = 0; t < t_steps; ++t)
        for (int32_t x = 0; x < x_steps; ++x)
            result.push_back(data[t][x]);

    double end = MPI_Wtime();
    dump(result, std::string{"data_linear.out"});

    std::cout << (end - start) << " " << 1 << std::endl;
}

std::vector<std::vector<double>> init() {
    std::vector<std::vector<double>> data{};
    data.resize(t_steps);

    for (auto&& data_it : data)
        data_it.resize(x_steps);
    
    for (int32_t x = 0; x < x_steps; ++x)
        data[0][x] = phi(x * h);

    for (int32_t t = 0; t < t_steps; ++t)
        data[t][0] = psi(t * tau);

    return data;
}

void dump(std::vector<double>& data, std::string&& out_f) {
    std::ofstream out{};
    out.open(out_f, std::ofstream::out);

    for (auto& data_it : data)
        out << data_it << std::endl;

    out.close();
}