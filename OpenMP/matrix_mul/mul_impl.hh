#ifndef MUL_IMPL
#define MUL_IMPL

#include <immintrin.h>
#include <omp.h>
//
#include "matrix.hh"

namespace mul_optimiz {

using type = int;
using Matr_int = Linear_space::Matrix<type>;

/**
 * @brief
 *
 */
namespace openmp_impl {

auto mul(const Matr_int &lhs, const Matr_int &rhs, bool is_simd = false) {
  auto lhs_rows = lhs.nrows(), rhs_cols = rhs.nclmns(), rhs_rows = rhs.nrows();

  size_t i = 0, j = 0, k = 0;

  Matr_int res{lhs_rows, rhs_cols};

  if (!is_simd) {
    //
#pragma omp parallel for private(i, j, k) shared(lhs, rhs, res)
    for (i = 0; i < lhs_rows; ++i)
      for (j = 0; j < rhs_cols; ++j)
        for (k = 0; k < rhs_rows; ++k)
          res[i][j] += lhs[i][k] * rhs[k][j];

  } else {

#pragma omp parallel for private(i, j, k) shared(lhs, rhs, res)
    for (i = 0; i < lhs_rows; ++i) {
      for (j = 0; j < rhs_cols; ++j) {
        int tmp = 0;
#pragma omp simd reduction(+ : tmp)
        for (k = 0; k < rhs_rows; ++k) {
          tmp += lhs[i][k] * rhs[k][j];
        }
        //
        res[i][j] += tmp;
      }
    }
  }

  return res;
}
} // namespace openmp_impl
/**
 * @brief
 *
 */
namespace simd_impl {

auto mul(const Matr_int &lhs, const Matr_int &rhs) {
  auto lhs_rows = lhs.nrows(), lhs_cols = lhs.nclmns(), rhs_cols = rhs.nclmns(),
       rhs_rows = rhs.nrows();
  Matr_int result{lhs_rows, rhs_cols};
  //
  uint optim_size = (rhs_cols - rhs_cols % 32);
  uint optim_rhs_col = rhs_cols / 32;

  assert(lhs_cols == rhs_rows);

  for (uint i = 0; i < lhs_rows; ++i) {
    __m256i *res = (__m256i *)result[i];

    for (uint j = 0; j < lhs_cols; ++j) {
      const __m256i *rhs_row = (__m256i *)rhs[j];
      __m256i lhs_elem = _mm256_set1_epi32(lhs[i][j]);

      int lhs_elem_ = lhs[i][j];
      const int *rhs_row_ = rhs[j];

      for (uint k = 0; k < optim_rhs_col; k += 1) {
        _mm256_storeu_si256(
            res + k * 4 + 0,
            _mm256_add_epi32(
                _mm256_mullo_epi32(lhs_elem,
                                   _mm256_loadu_si256(rhs_row + k * 4 + 0)),
                _mm256_loadu_si256(res + k * 4 + 0)));

        _mm256_storeu_si256(
            res + k * 4 + 1,
            _mm256_add_epi32(
                _mm256_mullo_epi32(lhs_elem,
                                   _mm256_loadu_si256(rhs_row + k * 4 + 1)),
                _mm256_loadu_si256(res + k * 4 + 1)));

        _mm256_storeu_si256(
            res + k * 4 + 2,
            _mm256_add_epi32(
                _mm256_mullo_epi32(lhs_elem,
                                   _mm256_loadu_si256(rhs_row + k * 4 + 2)),
                _mm256_loadu_si256(res + k * 4 + 2)));

        _mm256_storeu_si256(
            res + k * 4 + 3,
            _mm256_add_epi32(
                _mm256_mullo_epi32(lhs_elem,
                                   _mm256_loadu_si256(rhs_row + k * 4 + 3)),
                _mm256_loadu_si256(res + k * 4 + 3)));
      }

      for (uint k = optim_size; k < rhs_cols; ++k)
        result[i][k] += rhs_row_[k] * lhs_elem_;
    }
  }
  return result;
}
} // namespace simd_impl

} // namespace mul_optimiz

#endif //! MUL_IMPL