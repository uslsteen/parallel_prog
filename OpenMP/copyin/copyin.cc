#include <iostream>
#include <omp.h>

enum class Conds : uint8_t {
    ENABLE = 0,
    DISABLE = 1
};

int thread_data = 1;
#pragma omp threadprivate(thread_data)

void copyin_test(Conds condition) {
    thread_data = 1;
    #pragma omp parallel
    {
        thread_data = omp_get_thread_num();        
    }

    switch (condition)
    {
    case Conds::ENABLE: {
        #pragma omp parallel copyin(thread_data)
        {
        #pragma omp critical
            std::cout << "${thread_data} " << thread_data << " printed by thread = " << omp_get_thread_num() << std::endl; 
        }
        break;
    }
    case Conds::DISABLE: {
        #pragma omp parallel
        {
        #pragma omp critical
            std::cout << "${thread_data} " << thread_data << " printed by thread = " << omp_get_thread_num() << std::endl; 
        }
        break;
    }
    default:
        break;
    }
}

int main () {
    omp_set_num_threads(8);
    std::cout << "Copyin was enabled\n";
    copyin_test(Conds::ENABLE);
    //
    std::cout << "Copyin was disabled\n";
    copyin_test(Conds::DISABLE);
    //
    return 0;
}