// The rewritten CPU solver must reproduce the thesis code (tests/legacy_reference.hpp)
// step for step: same boundary conditions, same discretisation.

#include <algorithm>
#include <cmath>

#include "legacy_reference.hpp"
#include "tcc/cpu_solver.hpp"
#include "test_framework.hpp"

using namespace tcc;

namespace {

double max_abs(const std::vector<float>& a) {
    double m = 0.0;
    for (float x : a) m = std::max(m, std::fabs(static_cast<double>(x)));
    return m;
}

// Largest difference, relative to the largest magnitude in the reference.
double relative_difference(const std::vector<float>& got, const std::vector<float>& ref) {
    if (got.size() != ref.size()) return 1e30;
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - ref[i]));
    return worst / std::max(max_abs(ref), 1e-30);
}

void compare_with_legacy(const Config& cfg, int steps, int ipf) {
    const bool tube = cfg.type == CaseType::Tube;
    Config c = cfg;
    c.pressure_iterations = ipf;
    c.advection = Advection::Backward;  // the thesis scheme (the cavity preset defaults to upwind)

    CpuSolver solver(c);
    legacy::State ref(c.cells());
    for (int s = 0; s < steps; ++s) {
        solver.step();
        legacy::step(tube, c, ref, ipf);
    }
    REQUIRE(max_abs(ref.U) > 0.0);  // the comparison is meaningless on an all-zero field
    CHECK(relative_difference(solver.fields().u, ref.U) < 1e-5);
    CHECK(relative_difference(solver.fields().v, ref.V) < 1e-5);
    CHECK(relative_difference(solver.fields().p, ref.P) < 1e-4);
    CHECK(relative_difference(solver.fields().b, ref.B) < 1e-4);
}

}  // namespace

TEST(matches_thesis_tube_kernels_ipf1) {
    Config c = Config::tube();
    c.nx = 32;
    c.ny = 16;
    compare_with_legacy(c, 40, 1);
}

TEST(matches_thesis_tube_kernels_ipf7) {
    Config c = Config::tube();
    c.nx = 41;
    c.ny = 13;
    compare_with_legacy(c, 25, 7);
}

TEST(matches_thesis_cavity_cpu_code_ipf1) {
    Config c = Config::cavity();
    c.nx = c.ny = 24;
    compare_with_legacy(c, 40, 1);
}

TEST(matches_thesis_cavity_cpu_code_ipf9) {
    Config c = Config::cavity();
    c.nx = 30;
    c.ny = 18;
    compare_with_legacy(c, 25, 9);
}
