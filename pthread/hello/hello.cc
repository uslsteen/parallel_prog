#include <iostream>
#include <pthread.h>
#include <string>
#include <vector>

/* function to be run as a thread always must have the same signature:
   it has one void* parameter and returns void */
void *thr_fn(void *arg);

inline void usage(char* exec_path) {    
  std::cout << "USAGE: " << exec_path << " <nthreads>" << std::endl;
}

inline void check_err(int err_code) {
  if (!err_code)
    return;
  
  std::cerr << "Error was occured: "  << err_code;
  std::abort();
}

int main(int argc, char** argv) {
  if (argc == 2) {
    uint32_t nthreads = std::stoi(argv[1]);
    std::vector<pthread_t> threads(nthreads);
    std::vector<std::string> thr_msgs(nthreads);

    for (uint32_t thr_ind = 0; thr_ind < nthreads; ++thr_ind) {
      thr_msgs[thr_ind] = "Hello world: " + std::to_string(thr_ind);
      check_err(pthread_create(&threads[thr_ind], nullptr, thr_fn, (void*)thr_msgs[thr_ind].c_str()));
    }

    for (auto& thr : threads)
      check_err(pthread_join(thr, nullptr));

  }
  else usage(argv[0]);
  return 0;
}

void *thr_fn(void *arg) {
  std::cout << (char*) arg << std::endl;
  pthread_exit(NULL);
}