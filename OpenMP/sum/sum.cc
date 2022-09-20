#include <iostream>
#include <omp.h>


inline void usage(char* exec_path) {    
    std::cout << "USAGE:  ./sum" <<  exec_path << " <num>\n";
}

double calc_sum(int num, int thr_num, int thread_amount) {
    double cur_job = num / thread_amount;
    double data = 0;

    uint32_t cur_start = 1 + cur_job * (thr_num - 1);
    //
    //! NOTE: if cur_rank is last rank in communicator
    if (thr_num == thread_amount)
         cur_job += num % thread_amount;

    for (uint32_t i = cur_start, cur_end = cur_start + cur_job; i < cur_end; ++i)
        data += static_cast<double>(1) / i;

    return data;
}

int main(int argc, char** argv) {
    if (argc == 2) {
        auto num = std::atoi(argv[1]), thread_amount = omp_get_max_threads();
        double res = 0;
        // 
        #pragma omp parallel num_threads(thread_amount) reduction(+:res)
        {
            auto thr_num = omp_get_thread_num() + 1;
            res += calc_sum(num, thr_num, thread_amount);
        }
        std::cout << res << std::endl;
    }
    else
        usage(argv[0]);
    //
    return 0;
}