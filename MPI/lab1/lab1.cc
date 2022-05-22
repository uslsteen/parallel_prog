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
constexpr double h = 5e-3,
                 tau = 5e-3;

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

    std::vector<double> result{};
    result.resize(x_steps * t_steps);

    double start = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    do_main_job(data, rank, commsize);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = MPI_Wtime();

    result = do_last_part(data, rank, commsize);
    // result.resize(x_steps * t_steps);

    if (rank == 0) {
        std::cout << end - start << " " << commsize << std::endl;
        dump(result, std::string{"data_parallel.out"});
    }
}

void do_main_job(std::vector<std::vector<double>>& data, int32_t rank, int32_t commsize) {

    double cur_job = x_steps / commsize,
           part_job = x_steps % commsize;
    
    for (int32_t t = 0; t < t_steps - 1; ++t) {

        if (rank != commsize - 1) {
			for (int32_t x = rank * cur_job + 1; x < (rank + 1) * cur_job + 1; x++) {
					data[t + 1][x] = (f(t * tau, x * h) + 0.5 * tau / (h * h) * \
						(data[t][x + 1] - 2 * data[t][x] + data[t][x - 1]) + (data[t][x + 1] - data[t][x - 1])/ (2 * h)) * \
							tau + data[t][x];
			}

			data[t + 1][(rank + 1) * cur_job] = (f(t * tau, (x_steps - 1) * h) - (data[t][x_steps - 1] - data[t][x_steps - 2]) / h) * tau + data[t][x_steps - 1];
			
            if (rank != commsize - 2)
				MPI_Send(&data[t + 1][(rank + 1) * cur_job], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
		
			MPI_Recv(&data[t + 1][(rank + 1) * cur_job], 1, MPI_DOUBLE, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			if (rank != 0) {
				MPI_Recv(&data[t + 1][rank * cur_job], 1, MPI_DOUBLE, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(&data[t + 1][rank * cur_job + 1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
			}
			
            MPI_Send(&data[t + 1][rank * cur_job + 1], cur_job, MPI_DOUBLE, commsize - 1, 0, MPI_COMM_WORLD);
		}
        else if (rank == commsize - 1) {
            for (int32_t x = rank * cur_job + 1; x < x_steps - 1; x++) {
                    data[t + 1][x] = (f(t * tau, x * h) + 0.5 * tau / (h * h) * (data[t][x + 1] - 2 * data[t][x] + data[t][x - 1]) +
                                                                                (data[t][x + 1] - data[t][x - 1])/ (2 * h)) * tau + data[t][x];
            }

            if (commsize != 1)
                MPI_Send(&data[t + 1][rank * cur_job + 1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
                
            data[t + 1][x_steps - 1] = (f(t * tau, (x_steps - 1) * h) - (data[t][x_steps - 1] - data[t][x_steps - 2]) / h) * tau + data[t][x_steps - 1];

            for (int32_t i = 0; i < commsize - 1; i++)
                MPI_Recv(&data[t + 1][i * cur_job + 1], cur_job, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
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