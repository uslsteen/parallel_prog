#ifndef TREE_PROBLEM_MY_TIME_HPP
#define TREE_PROBLEM_MY_TIME_HPP

#include <chrono>

namespace timer {

using microsecs = std::chrono::microseconds;

class Timer final {
  using clock_t = std::chrono::high_resolution_clock;

  std::chrono::time_point<clock_t> beg;

public:
  Timer() : beg(clock_t::now()) {}

  void reset_time() { beg = clock_t::now(); }

  template <typename time_type> double elapsed() const {
    return std::chrono::duration_cast<time_type>(clock_t::now() - beg).count();
  }
};
} // namespace timer

#endif // TREE_PROBLEM_MY_TIME_HPP