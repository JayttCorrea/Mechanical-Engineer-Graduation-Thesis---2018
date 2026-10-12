#pragma once
// Run-time loading of the OpenCL library.
//
// The executables do not import OpenCL.dll: it is opened with LoadLibrary when the GPU
// solver is first needed. A PC without a graphics driver can therefore still start the
// programs and use the CPU solver, instead of failing with "OpenCL.dll was not found".

#include <CL/cl.h>

#include <string>

namespace tcc {

// Every OpenCL entry point the GPU solver uses.
#define TCC_CL_FUNCTIONS(X)        \
    X(clGetPlatformIDs)            \
    X(clGetPlatformInfo)           \
    X(clGetDeviceIDs)              \
    X(clGetDeviceInfo)             \
    X(clCreateContext)             \
    X(clCreateCommandQueue)        \
    X(clCreateProgramWithSource)   \
    X(clBuildProgram)              \
    X(clGetProgramBuildInfo)       \
    X(clCreateKernel)              \
    X(clCreateBuffer)              \
    X(clSetKernelArg)              \
    X(clEnqueueNDRangeKernel)      \
    X(clEnqueueFillBuffer)         \
    X(clEnqueueReadBuffer)         \
    X(clEnqueueWriteBuffer)        \
    X(clFinish)                    \
    X(clReleaseContext)            \
    X(clReleaseCommandQueue)       \
    X(clReleaseProgram)            \
    X(clReleaseKernel)             \
    X(clReleaseMemObject)

struct OpenClApi {
#define TCC_CL_DECLARE(name) decltype(&::name) name = nullptr;
    TCC_CL_FUNCTIONS(TCC_CL_DECLARE)
#undef TCC_CL_DECLARE

    bool available = false;  // library found and every function resolved
    std::string error;       // why not, when available == false
};

// Loaded once on first use; thread-safe (function-local static).
const OpenClApi& opencl();

}  // namespace tcc
