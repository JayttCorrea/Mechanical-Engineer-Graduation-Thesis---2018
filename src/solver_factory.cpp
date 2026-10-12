#include "tcc/solver_factory.hpp"

#include <stdexcept>

#include "tcc/cpu_solver.hpp"
#ifdef TCC_WITH_OPENCL
#include "tcc/gpu_solver.hpp"
#endif

namespace tcc {

const char* to_string(Backend backend) {
    return backend == Backend::Cpu ? "cpu" : "gpu";
}

bool parse_backend(const std::string& text, Backend& out) {
    if (text == "cpu") { out = Backend::Cpu; return true; }
    if (text == "gpu") { out = Backend::Gpu; return true; }
    return false;
}

bool gpu_available() {
#ifdef TCC_WITH_OPENCL
    return !list_opencl_devices().empty();
#else
    return false;
#endif
}

std::unique_ptr<Solver> make_solver(Backend backend, const Config& cfg) {
    if (backend == Backend::Cpu) return std::make_unique<CpuSolver>(cfg);
#ifdef TCC_WITH_OPENCL
    return std::make_unique<GpuSolver>(cfg);
#else
    throw std::runtime_error("this build has no OpenCL support (rebuild with TCC_BUILD_GPU=ON)");
#endif
}

}  // namespace tcc
