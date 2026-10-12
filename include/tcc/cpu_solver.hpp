#pragma once
// Reference solver running on the CPU. Serial unless built with TCC_OPENMP.

#include "tcc/field_guard.hpp"
#include "tcc/solver.hpp"

namespace tcc {

class CpuSolver final : public Solver {
public:
    explicit CpuSolver(const Config& cfg);

    std::string backend_name() const override;
    void reset() override;
    void step() override;
    void wait() override {}
    const Fields& fields() override;
    void set_velocity(const std::vector<float>& u, const std::vector<float>& v) override;
    void set_velocity_limit(float limit) override;

private:
    Fields fields_;
    std::vector<float> p_next_, u_next_, v_next_;  // ping-pong partners
    FieldGuard guard_u_, guard_v_, guard_p_, guard_b_;
};

}  // namespace tcc
