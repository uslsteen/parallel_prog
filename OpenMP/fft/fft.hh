#ifndef FFT_HH
#define FFT_HH

#include <bit>
#include <bitset>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>
//
#include <omp.h>
//
#include "timer.hh"

namespace fft {

/**
 * @brief Useful usings
 *
 */

using complex = std::complex<double>;
using compl_vec = std::vector<complex>;
//
using func = std::function<compl_vec(const compl_vec &)>;

constexpr double eps = 1e-6;

inline bool is_eq(complex fst, complex snd) { return std::abs(fst - snd) < eps; }

// template <typename T>
/**
 * @brief
 *
 * @param[in] data
 * @return compl_vec
 */
compl_vec naive_FFT(const compl_vec &data) {
  compl_vec result(data.size());
  complex pow = {0, -2.0 * M_PI / data.size()};

  for (std::size_t k = 0; k < result.size(); ++k) {
    auto &Xk = result[k];
    for (std::size_t j = 0; j < result.size(); ++j)
      Xk += data[j] * std::exp(pow * static_cast<double>(k * j));
  }
  return result;
}

/**
 * @brief
 *
 * @param[in] data
 * @return compl_vec
 */
compl_vec ct_FFT(const compl_vec &data) {
  auto N = data.size();

  if (N <= 1)
    return data;
  //
  auto half_N = N >> 1;
  //
  compl_vec even(half_N);
  compl_vec odd(half_N);

  for (size_t i = 0; i < half_N; ++i) {
    even[i] = data[2 * i];
    odd[i] = data[2 * i + 1];
  }

  auto res_even = ct_FFT(even);
  auto res_odd = ct_FFT(odd);

  compl_vec result(N);

  for (size_t i = 0; i < half_N; ++i) {
    auto t = std::exp(complex(0, -2 * M_PI * i / N)) * res_odd[i];
    result[i] = res_even[i] + t;
    result[i + half_N] = res_even[i] - t;
  }

  return result;
}

/**
 * @brief
 *
 * @param[in] val
 * @param[in] log2n
 * @return size_t
 */
size_t bit_reverse(size_t val, size_t log2n) {
  //
  std::size_t res = 0;
  for (std::size_t i = 0; i < log2n; ++i) {
    res <<= 1;
    res |= (val & 0x1);
    val >>= 1;
  }

  return res;
}

/**
 * @brief
 *
 * @param[in] data
 * @return compl_vec
 */
compl_vec ct_opt_FFT(const compl_vec &data) {
  std::complex<double> J = {0, 1};
  auto N = data.size();
  auto log2n = std::__bit_width(N) - 1;
  int n = 1 << log2n;

  compl_vec result(N);
  //
  for (unsigned int i = 0; i < n; ++i) {
    result[bit_reverse(i, log2n)] = data[i];
  }
  for (int stage = 1; stage <= log2n; ++stage) {
    int base = 1 << stage;
    int half_base = base >> 1;
    complex w(1, 0), wm = exp(-J * (M_PI / half_base));
    //
    for (int j = 0; j < half_base; ++j) {
      for (int k = j; k < n; k += base) {
        //
        std::complex<double> t = w * result[k + half_base];
        std::complex<double> u = result[k];
        result[k] = u + t;
        result[k + half_base] = u - t;
      }
      //
      w *= wm;
    }
  }
  return result;
}

/**
 * @brief 
 * 
 * @param[in] data 
 * @return compl_vec 
 */
compl_vec ct_par_FFT(const compl_vec &data) {
  auto N = data.size();

  if (N <= 1)
    return data;
  //
  auto half_N = N >> 1;
  //
  compl_vec even(half_N);
  compl_vec odd(half_N);

  for (std::size_t i = 0; i < half_N; ++i) {
    even[i] = data[2 * i];
    odd[i] = data[2 * i + 1];
  }
  //
  compl_vec res_even, res_odd;
  //
#pragma omp parallel
  {
#pragma omp single nowait
    {
#pragma omp task
      res_even = ct_par_FFT(even);
#pragma omp task
      res_odd = ct_par_FFT(odd);
    }
  }
  compl_vec result(N);

#pragma omp parallel for
  for (std::size_t i = 0; i < half_N; ++i) {
    auto t = std::exp(complex(0, -2 * M_PI * i / N)) * res_odd[i];
    result[i] = res_even[i] + t;
    result[i + half_N] = res_even[i] - t;
  }

  return result;
}

/**
 * @brief
 *
 * @param[in] data
 * @return compl_vec
 */
compl_vec ct_opt_par_FFT(const compl_vec &data) {
  std::complex<double> J = {0, 1};
  auto N = data.size();
  auto log2n = std::__bit_width(N) - 1;
  int n = 1 << log2n;

  compl_vec result(N);

  std::vector<std::complex<double>> res(N);
#pragma omp parallel for shared(res, data)
  for (int i = 0; i < n; ++i) {
    res[bit_reverse(i, log2n)] = data[i];
  }
  //
  for (int stage = 1; stage <= log2n; ++stage) {
    int base = 1 << stage;
    int half_base = base >> 1;
    complex w(1, 0), wm = exp(-J * (M_PI / half_base));
    //
#pragma omp parallel for shared(res)
    for (int j = 0; j < half_base; ++j) {
      for (int k = j; k < n; k += base) {
        std::complex<double> t = w * res[k + half_base];
        std::complex<double> u = res[k];
        res[k] = u + t;
        res[k + half_base] = u - t;
      }
      w *= wm;
    }
  }
  return res;
}

/**
 * @brief 
 * 
 */
std::vector<std::pair<func, std::string>> funcs{
    std::make_pair(naive_FFT, "Naive FFT"),
    std::make_pair(ct_FFT, "Cooley-Tukey FFT"),
    std::make_pair(ct_par_FFT, "Parallel Cooley-Tukey FFT"),
    std::make_pair(ct_opt_FFT, "Optimized Cooley-Tukey FFT"),
    std::make_pair(ct_opt_par_FFT, "Optimized parallel Cooley-Tukey FFT")};

} // namespace fft

#endif // FFT_HH