#include "opencl_loader.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tcc {

namespace {

using AnyFunction = void (*)();

OpenClApi load_opencl() {
    OpenClApi api;
#ifdef _WIN32
    // Installed by the graphics driver in System32.
    HMODULE lib = LoadLibraryA("OpenCL.dll");
    if (!lib) {
        api.error = "OpenCL.dll was not found: install or update the graphics driver to use the GPU";
        return api;
    }
    auto symbol = [lib](const char* name) {
        // FARPROC -> void(*)() is the documented way to silence -Wcast-function-type
        return reinterpret_cast<AnyFunction>(GetProcAddress(lib, name));
    };
#else
    void* lib = dlopen("libOpenCL.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libOpenCL.so", RTLD_NOW);
    if (!lib) {
        api.error = "libOpenCL.so was not found: install an OpenCL driver to use the GPU";
        return api;
    }
    auto symbol = [lib](const char* name) { return reinterpret_cast<AnyFunction>(dlsym(lib, name)); };
#endif

#define TCC_CL_LOAD(name)                                                     \
    api.name = reinterpret_cast<decltype(api.name)>(symbol(#name));          \
    if (!api.name) {                                                          \
        api.error = "the installed OpenCL library has no " #name " (OpenCL 1.2 is required)"; \
        return api;                                                           \
    }
    TCC_CL_FUNCTIONS(TCC_CL_LOAD)
#undef TCC_CL_LOAD

    api.available = true;
    return api;  // the library stays loaded for the life of the process
}

}  // namespace

const OpenClApi& opencl() {
    static const OpenClApi api = load_opencl();
    return api;
}

}  // namespace tcc
