/*
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp> 
//
#include <iostream>
#include <algorithm>
#include <vector>
//

class Driver {
private:
  cl::Context m_context;
  cl::Device m_device;

  void select_device() {
    std::vector<cl::Platform> pls;
    cl::Platform::get(&pls);

    std::cout << pls.size() << std::endl;

    for (auto &&pl_devs : pls) {
      std::vector<cl::Device> devs;
      pl_devs.getDevices(CL_DEVICE_TYPE_GPU, &devs);
      auto pred = [](const cl::Device &dev) {
        return dev.getInfo<CL_DEVICE_AVAILABLE>() &&
               dev.getInfo<CL_DEVICE_COMPILER_AVAILABLE>();
      };
      auto dev_it = std::find_if(devs.begin(), devs.end(), pred);
      if (dev_it != devs.end()) {
        m_device = *dev_it;
        return;
      }
    }

    throw std::runtime_error("Devices didn't find!\n");
  }

//
public:
  Driver() {
    select_device();
  }
};

//
int main() {
  Driver driver{};
  return 0;
}   
*/

#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main() {

  int i, j;
  char *value;
  size_t valueSize;
  cl_uint platformCount;
  cl_platform_id *platforms;
  cl_uint deviceCount;
  cl_device_id *devices;
  cl_uint maxComputeUnits;

  // get all platforms
  clGetPlatformIDs(0, NULL, &platformCount);
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformCount);
  clGetPlatformIDs(platformCount, platforms, NULL);

  for (i = 0; i < platformCount; i++) {

    // get all devices
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices,
                   NULL);

    // for each device print critical attributes
    for (j = 0; j < deviceCount; j++) {

      // print device name
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
      printf("%d. Device: %s\n", j + 1, value);
      free(value);

      // print hardware device version
      clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
      printf(" %d.%d Hardware version: %s\n", j + 1, 1, value);
      free(value);

      // print software driver version
      clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
      printf(" %d.%d Software version: %s\n", j + 1, 2, value);
      free(value);

      // print c version supported by compiler for device
      clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL,
                      &valueSize);
      value = (char *)malloc(valueSize);
      clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value,
                      NULL);
      printf(" %d.%d OpenCL C version: %s\n", j + 1, 3, value);
      free(value);

      // print parallel compute units
      clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                      sizeof(maxComputeUnits), &maxComputeUnits, NULL);
      printf(" %d.%d Parallel compute units: %d\n", j + 1, 4, maxComputeUnits);
    }

    free(devices);
  }

  free(platforms);
  return 0;
}