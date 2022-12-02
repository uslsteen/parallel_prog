#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
//
#include <omp.h>
//
#include "timer.hpp"

//
constexpr int JSIZE = 1000, ISIZE = 1000;

//
inline void usage(char *path) {
  std::cout << "USAGE : " << path << " num_of_process" << std::endl;
}

using data_t = std::vector<std::array<double, JSIZE>>;

void init(data_t &a) {
  for (size_t i = 0; i < ISIZE; ++i)
    for (size_t j = 0; j < JSIZE; ++j)
      a[i][j] = 10 * i + j;
}

data_t seq_impl(const data_t &a) {
  //
  data_t data{a};
  timer::Timer timer{};
  //
  for (size_t i = 0; i < ISIZE - 3; ++i)
    for (size_t j = 2; j < JSIZE; ++j)
      data[i][j] = std::sin(0.1 * data[i + 3][j - 2]);

  std::cout << "Seq. impl time : " << timer.elapsed<timer::microsecs>()
            << " microseconds" << std::endl;
  //
  return data;
}

data_t par_impl_both(const data_t &a) {
  //
  data_t data{a};
  timer::Timer timer{};

#pragma omp parallel for
  for (size_t i = 0; i < ISIZE - 3; ++i)
#pragma omp parallel for
    for (size_t j = 2; j < JSIZE; ++j)
      data[i][j] = std::sin(0.1 * a[i + 3][j - 2]);

  std::cout << "Par. impl time both : " << timer.elapsed<timer::microsecs>()
            << " microseconds" << std::endl;
  //
  return data;
}

data_t par_impl_inner(const data_t &a) {
  //
  data_t data{a};
  timer::Timer timer{};
  //
  for (size_t i = 0; i < ISIZE - 3; ++i)
#pragma omp parallel for
    for (size_t j = 2; j < JSIZE; ++j)
      data[i][j] = std::sin(0.1 * data[i + 3][j - 2]);

  std::cout << "Par. impl time inner : " << timer.elapsed<timer::microsecs>()
            << " microseconds" << std::endl;
  //
  return data;
}

data_t par_impl_outter(const data_t &a) {
  //
  data_t data{a};
  timer::Timer timer{};
  //
#pragma omp parallel for
  for (size_t i = 0; i < ISIZE - 3; ++i)
    for (size_t j = 2; j < JSIZE; ++j)
      data[i][j] = std::sin(0.1 * a[i + 3][j - 2]);

  std::cout << "Par. impl time outter : " << timer.elapsed<timer::microsecs>()
            << " microseconds" << std::endl;
  //
  return data;
}

void dump(const data_t &data, std::string &&descr) {
  std::cout << descr << std::endl;
  //
  for (auto &&data_it : data) {
    for (auto &&it : data_it)
      std::cout << it << " ";
    std::cout << std::endl;
  }
}

void compare(const data_t &lhs, const data_t &rhs, std::string &&descr) {
  if (std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end()))
    std::cout << "Equal\n";
  else {
    std::cout << "Not equal\n";
    std::cout << descr << std::endl;
    dump(lhs, std::string{"lhs"});
    dump(rhs, std::string{"rhs"});
  }
}

int main(int argc, char **argv) {
  //
  if (argc == 2) {
    data_t a(ISIZE);
    init(a);

    auto &&seq_res = seq_impl(a);

    omp_set_num_threads(8);
    //
    auto &&par_inner = par_impl_inner(a);
    auto &&par_outter = par_impl_outter(a);
    auto &&par_both = par_impl_both(a);
    //
    compare(seq_res, par_inner, std::string{"Inner impl."});
    compare(seq_res, par_outter, std::string{"Outter impl."});
    compare(seq_res, par_both, std::string{"Both impl."});
    //
  } else
    usage(argv[0]);
  //
  return 0;
}