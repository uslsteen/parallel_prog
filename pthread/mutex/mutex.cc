#include <algorithm>
#include <iostream>
#include <string>
#include <pthread.h>
#include <vector>

void *thr_mut(void *arg);

inline void usage(char* exec_path) {    
    std::cout << "USAGE: " << exec_path << " <nthreads>" << std::endl;
}

inline void check_err(int err_code) {
    if (!err_code)
        return;
    
    std::cerr << "Error was occured: " << err_code;
    std::abort();
}

inline uint32_t incr() {
    static int32_t id;
    return ++id;
}

/* shared data between threads */
double shared_data = 0;
pthread_mutex_t lock_x = PTHREAD_MUTEX_INITIALIZER;

int main(int argc, char** argv) {
    if (argc == 2) {
        uint32_t nthreads = std::stoi(argv[1]);
        std::vector<pthread_t> threads(nthreads);
        std::vector<int32_t> thr_id(nthreads);
        std::generate(thr_id.begin(), thr_id.end(), incr);

        for (auto& id : thr_id)
            std::cout << id << " ";
        std::cout << std::endl;

        for (uint32_t i = 0; i < nthreads; ++i)
            check_err(pthread_create(&threads[i], nullptr, thr_mut, (int32_t*)&(thr_id[i])));

        for (auto& thr : threads)
            check_err(pthread_join(thr, nullptr));
    }
    else usage(argv[0]);
}

void *thr_mut(void *arg) {
    int32_t *data = (int32_t *)arg;
    
    /* get mutex before modifying and printing shared_x */
    pthread_mutex_lock(&lock_x);
    shared_data += *data;
    std::cout << "shared_data = " << shared_data << std::endl;

    pthread_mutex_unlock(&lock_x);
    pthread_exit(NULL);
}