#pragma once
// OpenCL solver. All state lives on the device; the host only sees it when
// fields() is called. Only available when the library is built with TCC_WITH_OPENCL.

#include <memory>
#include <string>
#include <vector>

#include "tcc/solver.hpp"

namespace tcc {

struct GpuDeviceInfo {
    int index = 0;
    std::string platform;
    std::string name;
    bool is_gpu = false;
};

// Every OpenCL device the installed drivers expose (empty if none / no driver).
std::vector<GpuDeviceInfo> list_opencl_devices();

struct GpuOptions {
    int device_index = -1;          // index from list_opencl_devices(); -1 = first GPU
    bool allow_cpu_fallback = true; // use a CPU OpenCL device if there is no GPU
};

class GpuSolver final : public Solver {
public:
    // Throws std::runtime_error with a readable message (no device, build log, ...).
    explicit GpuSolver(const Config& cfg, const GpuOptions& options = {});
    ~GpuSolver() override;

    std::string backend_name() const override;
    std::string device_name() const;

    void reset() override;
    void step() override;
    void wait() override;
    const Fields& fields() override;
    void set_velocity(const std::vector<float>& u, const std::vector<float>& v) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tcc
