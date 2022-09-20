#include <iostream>
#include <omp.h>
#include <string>

int main() {
    auto thr_amount = omp_get_max_threads();
    uint32_t data = 0;
    #pragma omp parallel for ordered
        for (uint32_t thr_num = 0; thr_num < thr_amount; ++thr_num) {

            std::string msg{"Thread " + std::to_string(thr_num + 1) + " increment : "};
            #pragma omp ordered
            {
                msg += std::to_string(data++);
            }
            std::cout << msg << std::endl;
        }
    
    return 0;
}