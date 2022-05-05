#include <iostream>
#include <string>
#include <memory>
#include <pthread.h>
#include <vector>

void *thr_calc(void *arg);

inline void usage(char* exec_path) {    
    std::cout << "USAGE: " << exec_path << " <nthreads> <num>" << std::endl;
}

inline void check_err(int err_code) {
    if (!err_code)
        return;
    
    std::cerr << "Error was occured: " << err_code;
    std::abort();
}

int main(int argc, char** argv) {
    if (argc == 3) {
        uint32_t nthreads = std::stoi(argv[1]), num = std::stoi(argv[2]);
        std::vector<pthread_t> threads(nthreads);
        std::vector<std::array<uint32_t, 2>> pairs(nthreads);

        int32_t thr_job = num / nthreads, offset = num % nthreads;
        double sum = 0;

        for (uint32_t i = 0; i < nthreads; ++i) {
            auto& cur_pair = pairs[i];

            if (i == 0)
                cur_pair[0] = 0, 
                cur_pair[1] = thr_job + offset;
            else
                cur_pair[0] = offset + thr_job * i, 
                cur_pair[1] = cur_pair[0] + thr_job;

            check_err(pthread_create(&threads[i], nullptr, thr_calc, (int32_t*)&(cur_pair)));
        }

        for (auto& thr : threads) {
            double* thr_sum = nullptr;
            check_err(pthread_join(thr, (void**)&thr_sum));
            sum += *thr_sum;

            delete thr_sum;
        }

        std::cout << "Result: " << sum;
    }
    else usage(argv[0]);

    return 0;
}

void *thr_calc(void *arg) {
    double* cur_sum = new double;
    uint32_t* pair = (uint32_t*) arg;

    for (uint32_t i = pair[0] + 1; i <= pair[1]; ++i)
        *cur_sum += (double) 1 / i;

    pthread_exit((void*)cur_sum);
    return 0;
}