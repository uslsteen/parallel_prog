#ifndef __CL_MUL__
#define __CL_MUL__

//
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

//
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>
#define CL_HPP_TARGET_OPENCL_VERSION 300

//
#include "../../include/matrix/matrix.hh"

    using namespace Linear_space;

/**
 * @brief OpenCL driver class
 *
 */
namespace opencl_mul {

using type = int;
using Matr = Linear_space::Matrix<type>;

class Driver final {
private:
  cl::Context context_;
  cl::Device device_;

  cl::Program::Sources sources_;
  cl::CommandQueue queue_;
  cl::Program prog_;

  std::string src_code_;

  cl::Kernel naive_mul_;

private:
  bool build();

  bool kernel_exec(cl::Kernel kernel, cl::NDRange global_size,
                   cl::NDRange local_size);
  void gpu_timing(cl::Event &event);
  void Device_selection();

public:
  Driver();
  Driver(Driver const &) = delete;
  Driver &operator=(Driver const &) = delete;

  ~Driver() = default;

  Matr cl_mul(Matr &lhs, Matr &rhs);
};

Matr mul(Matr &lhs, Matr &rhs);
const char *err_what(cl_int err_code);

} // namespace  opencl_mul

#endif /* __CL_MUL__ */