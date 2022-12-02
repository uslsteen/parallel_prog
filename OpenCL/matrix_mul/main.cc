#include "cl_mul.hh"

//
template <typename Data>
void Buffer_init(Data &buffer, uint &num_of_rws, uint &num_of_clmns) {
  std::cin >> num_of_rws >> num_of_clmns;
  buffer.reserve(num_of_rws * num_of_clmns);
  //
  for (size_t i = 0; i < num_of_rws * num_of_clmns; ++i)
    std::cin >> buffer[i];
}

//
void check(opencl_mul::Matr &lhs, const opencl_mul::Matr &rhs) {
  if (lhs == rhs)
    std::cout << "Equal\n";
  else
    std::cout << "Not equal\n";
}

int main() {
  uint lhs_rows = 0, lhs_cols = 0, rhs_rows = 0, rhs_cols = 0;
  std::vector<int> lhs_buf, rhs_buf, np_buf;
  //
  Buffer_init(lhs_buf, lhs_rows, lhs_cols);
  Buffer_init(rhs_buf, rhs_rows, rhs_cols);
  Buffer_init(np_buf, lhs_rows, rhs_cols);
  //
  opencl_mul::Matr lhs{lhs_rows, lhs_cols, lhs_buf};
  opencl_mul::Matr rhs{rhs_rows, rhs_cols, rhs_buf};
  opencl_mul::Matr np_ref{lhs_rows, rhs_cols, np_buf};

  auto res = opencl_mul::mul(lhs, rhs);
  check(np_ref, res);

  return 0;
}