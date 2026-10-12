#include "tcc/solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tcc {

Solver::Solver(const Config& cfg) : cfg_(cfg) {
    const std::string problem = cfg_.validate();
    if (!problem.empty()) throw std::invalid_argument("invalid configuration: " + problem);
}

void Solver::set_pressure_iterations(int n) {
    cfg_.pressure_iterations = std::max(1, n);
}

void Solver::set_time_step(float dt) {
    if (!(dt > 0.0f) || !std::isfinite(dt)) throw std::invalid_argument("time step must be positive and finite");
    cfg_.dt = dt;
}

void Solver::set_advection(Advection scheme) {
    cfg_.advection = scheme;
}

void Solver::check_velocity_size(const std::vector<float>& u, const std::vector<float>& v) const {
    if (u.size() != cfg_.cells() || v.size() != cfg_.cells())
        throw std::invalid_argument("velocity arrays must have nx*ny elements");
}

}  // namespace tcc
