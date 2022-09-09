#include <iostream>
#include <omp.h>

int main() {
   #pragma omp parallel 
   {
      auto i = omp_get_thread_num(), max = omp_get_max_threads();
      std::cout << "Hello from thread [" << i + 1 << "]/[" << max << "]" << std::endl;
   }
   return 0;
}