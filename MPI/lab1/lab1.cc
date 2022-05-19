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

    do_main_job(data, rank, commsize);
    std::vector<double> result = do_last_part(data, rank, commsize);
    result.resize(x_steps * t_steps);

    if (rank == 0)
        dump(result, std::string{"data_parallel.out"});
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
    std::vector<double> result{};
    
    for (uint32_t t = 1; t < t_steps; ++t) {
        for (uint32_t x = 1; x < x_steps; ++x)
            data[t][x] =  data[t-1][x] - (tau / h) * (data[t-1][x] - data[t-1][x - 1]) + tau * f(x * h, (t - 1) * tau);
    }

    for (int32_t t = 0; t < t_steps; ++t)
        for (int32_t x = 0; x < x_steps; ++x)
            result.push_back(data[t][x]);

    dump(result, std::string{"data_linear.out"});
}

std::vector<std::vector<double>> init() {
    std::vector<std::vector<double>> data{};
    data.resize(t_steps);

    auto phi = [](double x) { return std::cos(PI * x); };
    auto psi = [](double t) { return std::exp(-t); };

    std::for_each (data.begin() , data.end() , [](std::vector<double>& x) {x.resize (x_steps); });
    std::for_each (data.begin() + 1 , data.end() , 
                   [i = 1 , &psi] (std::vector<double>& t) mutable { t[0] = psi (tau * i++); });

    std::generate (data[0].begin(), data[0].end(), 
                   [i = 0 , &phi] () mutable { return phi (h * i++); });

    return data;
}

void dump(std::vector<double>& data, std::string&& out_f) {
    std::ofstream out{};
    out.open(out_f, std::ofstream::out);

    for (auto& data_it : data)
        out << data_it << std::endl;

    out.close();
}