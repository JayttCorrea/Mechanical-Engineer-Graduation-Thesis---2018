#include "tcc/solver.hpp"

#include <algorithm>
#include <stdexcept>

namespace tcc {

Solver::Solver(const Config& cfg) : cfg_(cfg) {
    const std::string problem = cfg_.validate();
    if (!problem.empty()) throw std::invalid_argument("invalid configuration: " + problem);
}

void Solver::set_pressure_iterations(int n) {
    cfg_.pressure_iterations = std::max(1, n);
}

void Solver::check_velocity_size(const std::vector<float>& u, const std::vector<float>& v) const {
    if (u.size() != cfg_.cells() || v.size() != cfg_.cells())
        throw std::invalid_argument("velocity arrays must have nx*ny elements");
}

}  // namespace tcc
