#include <omp.h>
#include <iostream>

inline void usage(char* exec_path) {    
    std::cout << "USAGE:  ./nested" <<  exec_path << " <max_depth>\n";
}
//
void report_num_threads(int level) {
    #pragma omp single
    {
        printf("Level %d: number of threads in the team - %d\n",
               level, omp_get_num_threads());
    }
}
//
void nested(int cur_depth, int max_depth) {
    if (cur_depth == max_depth)
        return;

    auto thr_amount = omp_get_num_threads();
    #pragma omp parallel num_threads(thr_amount)
    {
        report_num_threads(cur_depth);
        nested(cur_depth + 1, max_depth);
    }
}
//
int main(int argc, char** argv) {
    if (argc == 2) {
        auto max_depth = std::stoi(argv[1]);
        //
        //! NOTE: set_dynamic - 
        omp_set_dynamic(0);
        omp_set_nested(1);
        nested(1, max_depth);
    }
    else
        usage(argv[0]);

    return 0;
}