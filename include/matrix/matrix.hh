#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cassert>
#include <cmath>
#include <ctime>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <vector>

namespace Linear_space {
template <typename Data> class Matrix final {
private:
  //! Num of rows and columns of my matrix
  uint rows = 0, clmns = 0;

  //! Perfomance of my matrix
  Data **matrix = nullptr;

  //! using for special data type of my matrix
  using DataIt = typename std::vector<Data>::iterator;

  //! Constant for copmaring
  const int EPSILON = 1e-6;

  struct Row_struct {
    uint cols = 0;
    Data *matr_row = nullptr;

    Row_struct(uint cols, Data *row) : cols(cols), matr_row(row) {}

    const Data &operator[](size_t i) const {
      assert(i < cols);
      return matr_row[i];
    }

    Data &operator[](size_t i) {
      assert(i < cols);
      return matr_row[i];
    }

    Row_struct(const Row_struct &row_m) = default;

    Row_struct &operator=(const Row_struct &row_m) = default;
  };

public:
  //! Constructors for class Matrix

  //! Constructor for matrix of zeros
  Matrix(int rows_, int clmns_) : rows(rows_), clmns(clmns_) {
    assert(rows_ * clmns_ != 0);

    matrix = new Data *[rows];
    for (size_t i = 0; i < rows; ++i) {
      matrix[i] = new Data[clmns];

      for (size_t j = 0; j < clmns; ++j)
        matrix[i][j] = 0;
    }
  }

  //! Constructor for matrix class of value
  Matrix(uint rows_, uint clmns_, Data val = Data{})
      : rows(rows_), clmns(clmns_) {
    assert(rows_ * clmns_ != 0);
    matrix = new Data *[rows];

    for (size_t i = 0; i < rows; ++i) {
      matrix[i] = new Data[clmns];

      for (size_t j = 0; j < clmns; ++j)
        matrix[i][j] = val;
    }
  }

  //! Constructor for matrix class from buffer
  Matrix(uint rows_, uint clmns_, const std::vector<Data> &buffer)
      : rows(rows_), clmns(clmns_) {
    assert(rows_ * clmns_ != 0);

    size_t num_of_elems = rows_ * clmns_, i = 0;

    matrix = new Data *[rows];

    for (i = 0; i < rows; ++i)
      matrix[i] = new Data[clmns];

    for (i = 0; i < num_of_elems; ++i)
      matrix[i / clmns][i % clmns] = buffer[i];
  }

  //! Constructor class matrix of two iterators to the vector
  Matrix(uint rows_, uint clmns_, const DataIt &beg, const DataIt &end)
      : rows(rows_), clmns(clmns_) {
    assert(rows_ * clmns_ != 0);

    DataIt current = beg;
    int num_of_elems = rows * clmns, cnter = 0, i = 0;

    matrix = new Data *[rows];

    for (i = 0; i < rows; ++i)
      matrix[i] = new Data[clmns];

    i = 0;

    for (DataIt cur = beg; i < num_of_elems; ++cur, ++i)
      matrix[i / clmns][i % clmns] = *cur;
  }

  //! Function returns a matrix of the upper triangular type
  static Matrix U_matr(uint num, Data elem) {
    Matrix matr{num, num};

    for (uint i = 0; i < num * num; ++i) {
      if ((i % num) == 0)
        for (int k = 0; k < (i / num); ++k)
          i++;

      uint rows_it = i / num;
      uint clmns_it = i % num;

      matr[rows_it][clmns_it] = elem;
    }

    return matr;
  }

  //! The function returns the identity matrix
  static Matrix eye(uint num) {
    Matrix matr(num, num);

    for (int i = 0; i < num; ++i)
      matr.matrix[i][i] = 1;

    return matr;
  }

  //! Copy constructor for class Matrix
  Matrix(const Matrix<int> &rhs) : rows(rhs.nrows()), clmns(rhs.nclmns()) {
    uint rhs_rows = rhs.nrows(), rhs_cols = rhs.nclmns();

    matrix = new Data *[rhs_rows];

    for (size_t i = 0; i < rhs_rows; ++i) {
      matrix[i] = new Data[rhs_cols];

      for (size_t j = 0; j < rhs_cols; ++j)
        matrix[i][j] = rhs[i][j];
    }
  }

  //! Destructor for matrix class
  ~Matrix() {
    for (size_t i = 0; i < rows; ++i)
      delete[] matrix[i];

    delete[] matrix;
    matrix = nullptr;
  }

  // Getters of class matrix
  uint nrows() const { return rows; }

  uint nclmns() const { return clmns; }

  //! Reloading of operators for class matrix

  Matrix<Data> &operator=(const Matrix<Data> &rhs) {
    if (rows != rhs.rows || clmns != rhs.clmns)
      Resize(rhs.rows, rhs.clmns);

    rows = rhs.rows;
    clmns = rhs.clmns;

    for (size_t i = 0; i < rows; ++i)

      for (size_t j = 0; j < clmns; ++j)
        matrix[i][j] = rhs.matrix[i][j];

    return (*this);
  }

  Matrix<Data> operator-() const {
    Matrix<Data> res_mtr{(*this)};

    for (int i = 0; i < rows; ++i)

      for (int j = 0; j < clmns; ++j)
        res_mtr.matrix[i][j] *= -1;

    return res_mtr;
  }

  Matrix<Data> &operator+=(const Matrix<Data> &mtr) {
    assert((rows == mtr.rows) && (clmns == mtr.clmns));

    rows = mtr.rows;
    clmns = mtr.clmns;

    for (size_t i = 0; i < rows; ++i)

      for (size_t j = 0; j < clmns; ++j)
        matrix[i][j] += mtr.matrix[i][j];

    return (*this);
  }

  Matrix<Data> &operator-=(const Matrix<Data> &mtr) {
    assert((rows == mtr.rows) && (clmns == mtr.clmns));

    rows = mtr.rows;
    clmns = mtr.clmns;

    for (size_t i = 0; i < rows; ++i)

      for (size_t j = 0; j < clmns; ++j)
        matrix[i][j] -= mtr.matrix[i][j];

    return (*this);
  }

  Matrix<Data> &operator*=(const Matrix<Data> &mtr) {
    assert(clmns == mtr.rows);

    Matrix tmp_mtr{rows, mtr.clmns, 0};

    for (std::size_t i = 0; i < rows; ++i)
      for (std::size_t j = 0; j < mtr.clmns; ++j)
        for (std::size_t k = 0; k < mtr.rows; ++k)
          tmp_mtr[i][j] += matrix[i][k] * mtr.matrix[k][j];

    *this = tmp_mtr;
    return (*this);
  }

  Matrix<Data> &operator*=(Data value) {
    assert((((*this).rows) * ((*this).clmns)) != 0);

    for (int i = 0; i < rows; ++i)

      for (int j = 0; j < clmns; ++j)
        matrix[i][j] *= value;

    return (*this);
  }

  Matrix<Data> &operator/=(Data value) {
    for (int i = 0; i < rows; ++i)

      for (int j = 0; j < clmns; ++j)
        matrix[i][j] /= value;

    return (*this);
  }

  bool operator==(const Matrix<Data> &mtr) {
    if (rows != mtr.rows && clmns != mtr.clmns)
      return false;

    for (size_t i = 0; i < rows; ++i)
      for (size_t j = 0; j < clmns; ++j)

        if (std::abs(mtr.matrix[i][j] - matrix[i][j]) > EPSILON)
          return false;

    return true;
  }

  Data *operator[](size_t i) const { return matrix[i]; }

  /*
  Row_struct operator [](size_t i) const
  {
      assert(i < rows);
      return Row_struct{clmns, matrix[i]};
  }
  Row_struct operator [](size_t i)
  {
      assert(i < rows);
      return Row_struct{clmns, matrix[i]};
  }
  */

private:
  //! Function for resing matrix
  void Resize(uint rows_, uint clmns_) {
    assert(rows_ * clmns_ != 0);

    for (size_t i = 0; i < rows; ++i)
      delete[] matrix[i];

    delete[] matrix;

    matrix = new Data *[rows_];

    for (size_t i = 0; i < rows_; ++i)
      matrix[i] = new Data[clmns_];
  }

public:
  //! Function for vectorization matrix
  void vectorize(std::vector<int> &mtr_buf) {
    // uint rows = mtr.nrows(), cols = mtr.nclmns();

    mtr_buf.reserve(rows * clmns);

    for (uint i = 0; i < rows; ++i)
      for (uint j = 0; j < clmns; ++j)

        mtr_buf.push_back(matrix[i][j]);
  }

  void set_zero() {
    for (uint i = 0; i < rows; ++i)
      for (uint j = 0; j < clmns; ++j)
        matrix[i][j] = 0;
  }

  //! Function for mul diagonal elements
  Data Diag_mul(const Matrix &mtr) {
    assert(mtr.rows == mtr.clmns);
    assert(mtr.matrix);
    Data res = 1;

    for (size_t i = 0; i < mtr.rows; ++i)
      res *= static_cast<double>(mtr.matrix[i][i]);

    return res;
  }

  //! Function for trannsposing matrix
  Matrix<Data> transposition() const {
    Matrix<Data> tmp_mtr{(*this)};
    Matrix<Data> res{clmns, rows};

    for (int i = 0; i < clmns; ++i)
      for (int j = 0; j < rows; ++j)
        res[i][j] = tmp_mtr[j][i];
    //
    return res;
  }

  //! Function for calculating determinant
  double determ() {
    assert(clmns == rows);
    assert(matrix);

    int swap_counter = 1;
    bool is_zero = false;

    Matrix<Data> tmp_mtr{(*this)};

    Gauss_algo(tmp_mtr, &swap_counter, &is_zero);

    if (is_zero)
      return 0;

    Data res = swap_counter * Diag_mul(tmp_mtr);

    return res;
  }

  //! Dump for matrix
  void dump(std::ostream &os) const {
    std::cout << std::endl;

    for (size_t i = 0; i < rows; ++i) {
      os << "|| ";

      for (size_t j = 0; j < clmns; ++j)
        os << matrix[i][j] << " ";

      os << "||" << std::endl;
    }
    std::cout << "Num of rows = " << rows << std::endl;
    std::cout << "Num of clmns = " << clmns << std::endl;

    std::cout << std::endl;
  }

  //! Functions for working with matrix rows

  //! Function for swapping matrix rows
  void swap_rows(uint i, uint j) {
    assert(i >= 0 && j >= 0);
    assert(i < (*this).rows && j < (*this).rows);

    std::swap((*this).matrix[i], (*this).matrix[j]);
  }

  //! Function for add matrix rows
  void sum_rows(uint i, uint j) {
    assert(i >= 0 && j >= 0);
    assert(i < (*this).rows && j < (*this).rows);

    for (int k = 0; k < (*this).clmns; ++k)
      (*this).matrix[k][i] += (*this).matrix[k][j];
  }

  //! Function for mul matrix rows
  void mul_rows_to_sclr(uint i, Data value) {
    assert(i >= 0);
    assert(i < (*this).rows);

    for (int k = 0; k < (*this).clmns; ++k)
      (*this).matrix[k][i] *= static_cast<double>(value);
  }
};

//! Also binary reloaded operators for working with matrix

template <typename Data>
std::ostream &operator<<(std::ostream &os, Matrix<Data> &matr) {
  for (size_t i = 0; i < matr.nrows(); ++i) {
    // os << "|| ";

    for (size_t j = 0; j < matr.nclmns(); ++j)
      os << matr[i][j] << " ";

    os << std::endl;
  }

  return os;
}

template <typename Data>
Matrix<Data> operator+(const Matrix<Data> &lhs, const Matrix<Data> &rhs) {
  Matrix<Data> res{lhs};

  res += rhs;
  return res;
}

template <typename Data>
Matrix<Data> operator-(const Matrix<Data> &lhs, const Matrix<Data> &rhs) {
  Matrix<Data> res{lhs};

  res -= rhs;
  return res;
}

template <typename Data>
Matrix<Data> operator*(const Matrix<Data> &lhs, const Matrix<Data> &rhs) {
  Matrix<Data> res{lhs};

  res *= rhs;
  return res;
}
/*
template <typename Data>
bool operator==(const Matrix<Data> &lhs, const Matrix<Data> &rhs) {
  return lhs == rhs;
}
*/
} // namespace Linear_space

#endif // MATRIX_MATRIX_H