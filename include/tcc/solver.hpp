#pragma once
// Backend-independent solver interface.

#include <cstdint>
#include <string>
#include <vector>

#include "tcc/config.hpp"
#include "tcc/fields.hpp"

namespace tcc {

class Solver {
public:
    // Throws std::invalid_argument if cfg.validate() reports a problem.
    explicit Solver(const Config& cfg);
    virtual ~Solver() = default;
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    const Config& config() const { return cfg_; }
    std::uint64_t steps() const { return steps_; }

    // How many NaN / infinite values have been replaced by the last valid value of their
    // cell since the last reset. Non-zero means the explicit scheme went unstable (the
    // results are then no longer a faithful picture of a real fluid).
    std::uint64_t repaired_values() const { return repaired_; }

    // Change the number of Jacobi sweeps per step (values < 1 are clamped to 1).
    void set_pressure_iterations(int n);

    // Change the time step or the convection scheme of a running simulation. Throws
    // std::invalid_argument for a time step that is not positive and finite.
    void set_time_step(float dt);
    void set_advection(Advection scheme);

    // Velocities above `limit` (in absolute value) are treated like NaN when the fields are
    // read back: they are replaced by the last valid value. 0 or infinity disables the limit
    // (the default: only NaN / infinity are repaired).
    virtual void set_velocity_limit(float limit) = 0;

    virtual std::string backend_name() const = 0;

    // Zero velocity and pressure, step counter back to 0.
    virtual void reset() = 0;
    // Advance one time step. May return before the work has finished (GPU).
    virtual void step() = 0;
    // Block until every queued step has finished.
    virtual void wait() = 0;
    // Host copy of the current state (downloads from the device if needed).
    // Non-finite values are repaired before they are returned (see repaired_values()).
    virtual const Fields& fields() = 0;
    // Overwrite the velocity field. Throws std::invalid_argument on a size mismatch.
    virtual void set_velocity(const std::vector<float>& u, const std::vector<float>& v) = 0;

protected:
    void check_velocity_size(const std::vector<float>& u, const std::vector<float>& v) const;

    Config cfg_;
    std::uint64_t steps_ = 0;
    std::uint64_t repaired_ = 0;
};

}  // namespace tcc
