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


/*
auto mul_n_transpose(const Matr_int &lhs, const Matr_int &rhs) {
  Matr_int rhs_t{rhs.transposition()};
  //
  Matr_int res{lhs.nrows(), rhs.nclmns()};

  std::size_t res_c = res.nclmns(), res_r = res.nrows(),
              com_sz = lhs.nclmns(), unroll_size = com_sz - com_sz % 16;

  for (std::size_t i = 0; i < res_r; ++i)
    for (std::size_t j = 0; j < res_c; ++j) {
      //
      auto lptr = lhs[i];
      auto rptr = rhs_t[j];
      std::size_t k = 0;
      for (; k < unroll_size; k += 16)
        res[i][j] += lptr[k] * rptr[k] + lptr[k + 1] * rptr[k + 1] +
                     lptr[k + 2] * rptr[k + 2] + lptr[k + 3] * rptr[k + 3] +
                     lptr[k + 4] * rptr[k + 4] + lptr[k + 5] * rptr[k + 5] +
                     lptr[k + 6] * rptr[k + 6] + lptr[k + 7] * rptr[k + 7] +
                     lptr[k + 8] * rptr[k + 8] + lptr[k + 9] * rptr[k + 9] +
                     lptr[k + 10] * rptr[k + 10] + lptr[k + 11] * rptr[k + 11] +
                     lptr[k + 12] * rptr[k + 12] + lptr[k + 13] * rptr[k + 13] +
                     lptr[k + 14] * rptr[k + 14] + lptr[k + 15] * rptr[k + 15];

      for (; k < com_sz; ++k)
        res[i][j] += lptr[k] * rptr[k];
    }

  return res;
}
*/