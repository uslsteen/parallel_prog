#include "fft.hh"

void input_data(fft::compl_vec &vec, std::istream &str = std::cin) {
  size_t size = 0;
  str >> size;
  vec.resize(size);
  for (auto &it : vec)
    str >> it;
}

void test(fft::compl_vec &x, fft::compl_vec &ref, std::pair<fft::func, std::string> func) {
  timer::Timer timer{};
  auto &&result = func.first(x);
  std::cout << func.second << "time : " << timer.elapsed<std::chrono::microseconds>() << " microsecs" << std::endl;

  if (std::equal(result.begin(), result.end(), ref.begin(), ref.end(),
                 fft::is_eq))
    std::cout << "Passed\n";
  else
    std::cout << "Failed\n";
  //
  std::cout << std::endl;
}

int main() {
  omp_set_num_threads(8);
  fft::compl_vec x{}, ref{};
  input_data(x);
  input_data(ref);
  //
  for (auto &f : fft::funcs)
    test(x, ref, f);
  //
  return 0;
}