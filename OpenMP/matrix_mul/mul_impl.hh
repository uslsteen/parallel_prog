#ifndef MUL_IMPL
#define MUL_IMPL

#include <immintrin.h>
#include <omp.h>
//
#include "matrix.hh"

namespace mul_optimiz {

using type = int;
using Matr_int = Linear_space::Matrix<type>;

auto vinograd_mul(const Matr_int &lhs, const Matr_int &rhs) {
  auto lhs_rows = lhs.nrows(), rhs_cols = rhs.nclmns(), rhs_rows = rhs.nrows();
  Matr_int res{lhs_rows, rhs_cols};

  int *proxy_row = new int[lhs_rows];
  int *proxy_col = new int[rhs_rows];

  std::size_t d = (lhs_rows) / 2;

  auto start = omp_get_wtime();

  for (std::size_t row = 0; row < lhs_rows; ++row) {
    proxy_row[row] = lhs[row][0] * lhs[row][1];
    for (std::size_t col = 1; col < d; ++col)
      proxy_row[row] += lhs[row][2 * col] * lhs[row][2 * col + 1];
  }

  for (std::size_t col = 0; col < rhs_cols; ++col) {
    proxy_col[col] = rhs[0][col] * rhs[1][col];
    for (std::size_t row = 1; row < d; ++row)
      proxy_col[col] += rhs[2 * row][col] * rhs[2 * row + 1][col];
  }

  for (std::size_t row = 0; row < lhs_rows; ++row) {
    for (std::size_t col = 0; col < rhs_cols; ++col) {
      res[row][col] = -proxy_row[row] - proxy_col[col];

      for (std::size_t k = 0; k < d; ++k) {
        res[row][col] += (lhs[row][2 * k] + rhs[2 * k + 1][col]) *
                         (lhs[row][2 * k + 1] + rhs[2 * k][col]);
      }
    }
  }
  auto end = omp_get_wtime();
  std::cout << "Vinograd linear time : " << end - start << std::endl;

  if (2 * d != lhs_rows) {
    for (std::size_t row = 0; row < lhs_rows; ++row) {
      for (std::size_t col = 0; col < rhs_cols; ++col)
        res[row][col] += lhs[row][rhs_rows - 1] * rhs[lhs_rows - 1][col];
    }
  }

  delete[] proxy_row;
  delete[] proxy_col;
  //
  return res;
}

/**
 * @brief
 *
 */
namespace openmp_impl {

auto vinograd_mul(const Matr_int &lhs, const Matr_int &rhs) {
  auto lhs_rows = lhs.nrows(), rhs_cols = rhs.nclmns(), rhs_rows = rhs.nrows();
  std::size_t row = 0, col = 1;
  Matr_int res{lhs_rows, rhs_cols};

  int *proxy_row = new int[lhs_rows];
  int *proxy_col = new int[rhs_rows];

  std::size_t d = (lhs_rows) / 2;

  #pragma omp parallel for private(row, col) shared(lhs, rhs, proxy_row)
  for (row = 0; row < lhs_rows; row++) {
    proxy_row[row] = lhs[row][0] * lhs[row][1];
    for (col = 1; col < d; col++)
      proxy_row[row] += lhs[row][2 * col] * lhs[row][2 * col + 1];
  }

  #pragma omp parallel for private(row, col) shared(lhs, rhs, proxy_col)
  for (col = 0; col < rhs_rows; col++) {
    proxy_col[col] = rhs[0][col] * rhs[1][col];
    for (row = 1; row < d; row++)
      proxy_col[col] += rhs[2 * row][col] * rhs[2 * row + 1][col];
  }

  #pragma omp parallel for private(row, col) shared(lhs, rhs, proxy_row, proxy_col, res)
  for (row = 0; row < lhs_rows; row++) {
    for (col = 0; col < rhs_cols; col++) {
      res[row][col] = -proxy_row[row] - proxy_col[col];

      for (std::size_t k = 0; k < d; k++) {
        res[row][col] += (lhs[row][2 * k] + rhs[2 * k + 1][col]) *
                         (lhs[row][2 * k + 1] + rhs[2 * k][col]);
      }
    }
  }

  if (2 * d != lhs_rows) {
  #pragma omp parallel for private(row, col) shared(lhs, rhs, res)
    for (row = 0; row < lhs_rows; row++) {
      for (col = 0; col < rhs_cols; col++)
        res[row][col] += lhs[row][rhs_rows - 1] * rhs[lhs_rows - 1][col];
    }
  }
  //
  delete[] proxy_row;
  delete[] proxy_col;
  //
  return res;
}

auto naive_mul(const Matr_int &lhs, const Matr_int &rhs, bool is_simd = false) {
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