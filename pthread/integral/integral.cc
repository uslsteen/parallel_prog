#include <iostream>
#include <thread>
#include <string>
#include <functional>
#include <limits>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

#include "thread_wrapper.hh"
#include "timer.hh"

const int SPLITS = 8;

const double START = -4.995;
const double END = 0; 

double GLOBAL_OFFSET = (END - START) / SPLITS;

inline void usage(char* path) {
    std::cout << "USAGE:" << path << " [nthreads] [accuracy]" << std::endl;
}

double simpson_meth(const std::function<double(double)> &fn, double a, double b, double h) {
    double local_integral = 0;
    size_t N = (b - a) / h + 1;

    for (size_t i = 1; i < N - 1; ++i)
        local_integral += fn(a + h * (i - 1)) + 4 * fn(a + h * i) + fn(a + h * (i + 1));

    local_integral *= h / 6.0;
    return local_integral;
}

double thread_calc(const std::function<double(double)> &fn,
                double a, double b, double h, double eps,
                threads_holder<double> &holder) {
    auto I = simpson_meth(fn, a, b, h);
    auto I2 = simpson_meth(fn, a, b, h / 4.0);

    //! NOTE: Runge error estimation
    auto delta = (I - I2) / 6.0;
    auto res = I2;
    
    if (std::abs(delta) > eps) {

        //! NOTE: divide the task into parts for every process and put it into queue
        double offset = (b - a) / SPLITS;

        for (size_t i = 0; i < SPLITS; ++i) {
            double cur_start = a + offset * i;
            double cur_end = a + offset * (i + 1);
            holder.add_job(thread_calc, fn, cur_start, cur_end, h / 4.0, eps, std::ref(holder));
        }

        res = 0.0;
    }
    return res;
}

int main(int argc, char** argv) {
    
    if (argc == 3) {
        size_t nthreads = std::stoll(argv[1]);
        double eps = std::stod(argv[2]);

        //! NOTE: versatility choice of the function
        auto&& fn = [](double x) { return std::sin(1/(5 + x)); };
        //! NOTE: also have tested:
        // auto&& fn = [](double x) { return std::cos(1/x)/x; };
        // auto&& fn = [](double x) { return std::sin(1/x); };


        //! NOTE: every thread will add local result to own var
        //! NOTE: finally, all local results by every threads will sum up
        std::vector<double> thr_local_res(nthreads);

        std::function<void(size_t, double)> results_sum = [&thr_local_res](size_t thr_id, double thr_res) -> void {
            thr_local_res[thr_id] += thr_res;
        };

        //! NOTE: init thread holder - object to hold and run threads job
        threads_holder<double> holder{nthreads, results_sum};

        double h = eps;
        Timer timer{};

        for (size_t i = 0; i < SPLITS; ++i) {

            double cur_start = START + GLOBAL_OFFSET * i, 
                   cur_end =   START + GLOBAL_OFFSET * (i + 1);

            holder.add_job(thread_calc, fn, cur_start, cur_end, h, eps / SPLITS, std::ref(holder));
        }

        holder.wait_all();
        auto real_time = timer.elapsed();

        // summarize the thr_local_res of each stream and get the final result
        auto res_integral = std::accumulate(thr_local_res.begin(), thr_local_res.end(), 0.0);

        std::cout << "time = " << real_time << std::endl;
        std::cout << "result = " << std::setprecision(8) <<  res_integral << std::endl;
        // std::cout << "num tasks = " << holder.last_job_id() << std::endl;
    }
    else usage(argv[0]);
    
    return 0;
}