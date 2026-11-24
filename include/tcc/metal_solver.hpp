#pragma once
// Metal solver. All state lives on the device; the host only sees it when
// fields() is called. Only available when the library is built with TCC_WITH_METAL.

#include <memory>
#include <string>
#include <vector>

#include "tcc/solver.hpp"

namespace tcc {

struct MetalDeviceInfo {
    int index = 0;
    std::string platform;
    std::string name;
    bool is_gpu = false;
};

// The default unified-memory Metal device (empty if unavailable).
std::vector<MetalDeviceInfo> list_metal_devices();

struct MetalOptions {
    int device_index = -1;          // index from list_metal_devices(); -1 = first GPU
};

class MetalSolver final : public Solver {
public:
    // Throws std::runtime_error with a readable message (no device, build log, ...).
    explicit MetalSolver(const Config& cfg, const MetalOptions& options = {});
    ~MetalSolver() override;

    std::string backend_name() const override;
    std::string device_name() const;

    void reset() override;
    void step() override;
    void wait() override;
    const Fields& fields() override;
    void set_velocity(const std::vector<float>& u, const std::vector<float>& v) override;
    void set_velocity_limit(float limit) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tcc
