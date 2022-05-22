#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

#include <omp.h>

inline double func(double x) {
  return sqrt(4 - x * x);
}

inline void usage(char* path) {
    std::cout << "USAGE:" << path << "[num of splits] [process_num]" << std::endl; 
}

int main(int argc, char** argv) {

    if (argc == 3) {
        int32_t N = std::stoi(argv[1]);
        int32_t num_threads = std::stoi(argv[2]);

        double value = 0;
        double x = 2.0/N;
        double start_time = omp_get_wtime();

        omp_set_num_threads(num_threads);
        #pragma omp parallel reduction(+:value)
        {
            int32_t size = omp_get_num_threads();
            int32_t rank = omp_get_thread_num();
            for (int32_t i = rank; i < N; i += size) {
                value += x * func(i * x);
        }
        }

        double end_time = omp_get_wtime();
        std::cout << "Time=" << end_time - start_time << " secs" << std::endl;
        std::cout << "I = " <<  value << std::endl;
    }
    else usage(argv[0]);

    return 0;
    
}
