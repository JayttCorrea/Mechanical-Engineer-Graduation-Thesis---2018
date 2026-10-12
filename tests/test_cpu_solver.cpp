#include <cmath>
#include <memory>

#include "tcc/cpu_solver.hpp"
#include "test_framework.hpp"

using namespace tcc;

namespace {

Config small_cavity(int n = 24) {
    Config c = Config::cavity();
    c.nx = c.ny = n;
    return c;
}

Config small_tube(int nx = 40, int ny = 16) {
    Config c = Config::tube();
    c.nx = nx;
    c.ny = ny;
    return c;
}

std::size_t at(const Config& c, int x, int y) {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(c.nx) + static_cast<std::size_t>(x);
}

// L2 norm of the discrete Poisson residual on interior cells (no boundary ghosts involved):
//   dy^2 (P_E + P_W) + dx^2 (P_N + P_S) - C P - C B = 0 with C = 2 (dx^2 + dy^2)
double poisson_residual(const Config& c, const Fields& f) {
    const double dx = c.dx(), dy = c.dy();
    const double cp = 2.0 * (dx * dx + dy * dy);
    double sum = 0.0;
    for (int y = 1; y < c.ny - 1; ++y) {
        for (int x = 1; x < c.nx - 1; ++x) {
            const std::size_t i = at(c, x, y);
            const double r = dy * dy * (static_cast<double>(f.p[i + 1]) + f.p[i - 1]) +
                             dx * dx * (static_cast<double>(f.p[i + c.nx]) + f.p[i - c.nx]) -
                             cp * f.p[i] - cp * f.b[i];
            sum += r * r;
        }
    }
    return std::sqrt(sum);
}

}  // namespace

TEST(cpu_starts_at_rest_and_counts_steps) {
    CpuSolver s(small_cavity());
    CHECK(s.steps() == 0);
    for (float x : s.fields().u) CHECK(x == 0.0f);
    s.step();
    s.step();
    CHECK(s.steps() == 2);
    s.reset();
    CHECK(s.steps() == 0);
    for (float x : s.fields().u) CHECK(x == 0.0f);
    for (float x : s.fields().p) CHECK(x == 0.0f);
}

TEST(cpu_fluid_at_rest_with_no_drive_stays_at_rest) {
    for (CaseType type : {CaseType::Tube, CaseType::Cavity}) {
        Config c = type == CaseType::Tube ? small_tube() : small_cavity();
        c.u0 = 0.0f;
        c.v0 = 0.0f;
        CpuSolver s(c);
        for (int i = 0; i < 20; ++i) s.step();
        for (float x : s.fields().u) CHECK(x == 0.0f);
        for (float x : s.fields().v) CHECK(x == 0.0f);
        for (float x : s.fields().p) CHECK(x == 0.0f);
    }
}

TEST(cpu_uniform_flow_is_untouched_away_from_the_walls) {
    // Plug flow u = u0, v = 0 matches the inlet and has no interior gradients, so only
    // the cells next to the no-slip walls may change in the first step.
    Config c = small_tube(40, 16);
    CpuSolver s(c);
    s.set_velocity(std::vector<float>(c.cells(), c.u0), std::vector<float>(c.cells(), 0.0f));
    s.step();
    const Fields& f = s.fields();
    for (int y = 2; y <= c.ny - 3; ++y) {
        for (int x = 0; x < c.nx; ++x) {
            CHECK(f.u[at(c, x, y)] == c.u0);
            CHECK(f.v[at(c, x, y)] == 0.0f);
        }
    }
    // ...and the wall rows do feel the no-slip condition.
    CHECK(f.u[at(c, c.nx / 2, 0)] < c.u0);
    CHECK(f.u[at(c, c.nx / 2, c.ny - 1)] < c.u0);
}

TEST(cpu_cavity_lid_drags_the_fluid) {
    Config c = small_cavity(32);
    CpuSolver s(c);
    for (int i = 0; i < 300; ++i) s.step();
    const Fields& f = s.fields();
    REQUIRE(all_finite(f.u) && all_finite(f.v) && all_finite(f.p));

    const int mid = c.nx / 2;
    const float top = f.u[at(c, mid, c.ny - 1)];
    const float below = f.u[at(c, mid, c.ny - 4)];
    const float centre = f.u[at(c, mid, c.ny / 2)];
    CHECK(top > 0.0f);
    CHECK(top <= 1.5f * c.u0);        // never much faster than the lid
    CHECK(top > below);               // momentum decays away from the lid
    CHECK(std::fabs(centre) < 0.1f * top);  // 3 s is far too short to reach the cavity centre
    CHECK(finite_range(f.v).max != finite_range(f.v).min);  // the cavity develops a v component
}

TEST(cpu_tube_flow_develops_from_the_inlet) {
    // A short tube so the inlet front (u0 * t = 0.04 m after 400 steps) crosses many cells.
    Config c = small_tube(60, 20);
    c.length_x = 0.2f;
    CpuSolver s(c);
    for (int i = 0; i < 400; ++i) s.step();
    const Fields& f = s.fields();
    REQUIRE(all_finite(f.u) && all_finite(f.v) && all_finite(f.p));
    // Centreline velocity near the inlet follows the inlet speed.
    CHECK_NEAR(f.u[at(c, 1, c.ny / 2)], c.u0, 0.2 * c.u0);
    // No-slip: the wall rows are slower than the centreline.
    CHECK(f.u[at(c, c.nx / 2, 0)] < f.u[at(c, c.nx / 2, c.ny / 2)]);
}

TEST(cpu_more_jacobi_sweeps_reduce_the_poisson_residual) {
    const Config base = small_cavity(24);
    double residual[3] = {};
    const int sweeps[3] = {1, 20, 300};
    for (int k = 0; k < 3; ++k) {
        CpuSolver s(base);
        for (int i = 0; i < 20; ++i) s.step();  // build up a non-trivial velocity field
        s.set_pressure_iterations(sweeps[k]);
        s.step();
        residual[k] = poisson_residual(base, s.fields());
    }
    CHECK(residual[0] > 0.0);
    CHECK(residual[1] < residual[0]);
    CHECK(residual[2] < residual[1]);
}

TEST(cpu_simulation_is_deterministic) {
    Config c = small_tube(30, 12);
    CpuSolver a(c), b(c);
    for (int i = 0; i < 50; ++i) { a.step(); b.step(); }
    CHECK(a.fields().u == b.fields().u);
    CHECK(a.fields().v == b.fields().v);
    CHECK(a.fields().p == b.fields().p);
}

TEST(cpu_set_velocity_validates_sizes_and_replaces_the_field) {
    Config c = small_cavity(10);
    CpuSolver s(c);
    CHECK_THROWS(s.set_velocity(std::vector<float>(5), std::vector<float>(c.cells())));
    CHECK_THROWS(s.set_velocity(std::vector<float>(c.cells()), std::vector<float>(5)));
    std::vector<float> u(c.cells(), 0.25f), v(c.cells(), -0.5f);
    s.set_velocity(u, v);
    CHECK(s.fields().u == u);
    CHECK(s.fields().v == v);
}

TEST(cpu_pressure_iterations_are_clamped_to_one) {
    CpuSolver s(small_cavity(8));
    s.set_pressure_iterations(0);
    CHECK(s.config().pressure_iterations == 1);
    s.set_pressure_iterations(-7);
    CHECK(s.config().pressure_iterations == 1);
    s.set_pressure_iterations(12);
    CHECK(s.config().pressure_iterations == 12);
}

TEST(cpu_warm_start_stays_finite) {
    Config c = small_cavity(20);
    c.warm_start_pressure = true;
    c.pressure_iterations = 5;
    CpuSolver s(c);
    for (int i = 0; i < 200; ++i) s.step();
    CHECK(all_finite(s.fields().u));
    CHECK(all_finite(s.fields().v));
    CHECK(all_finite(s.fields().p));
}

TEST(cpu_fills_a_non_square_mesh) {
    // nx != ny and neither a power of two: catches row/column mix-ups.
    Config c = small_tube(37, 11);
    CpuSolver s(c);
    for (int i = 0; i < 10; ++i) s.step();
    CHECK(s.fields().u.size() == 37u * 11u);
    CHECK(all_finite(s.fields().u));
}

TEST(cpu_upwind_advection_keeps_the_cavity_stable) {
    Config c = small_cavity(24);
    c.advection = Advection::Upwind;
    c.pressure_iterations = 20;
    CpuSolver s(c);
    for (int i = 0; i < 3000; ++i) s.step();
    const Fields& f = s.fields();
    REQUIRE(all_finite(f.u) && all_finite(f.v) && all_finite(f.p));
    CHECK(std::fabs(finite_range(f.u).max) <= 1.5f * c.u0);
    CHECK(std::fabs(finite_range(f.u).min) <= 1.5f * c.u0);
}

TEST(cpu_thesis_backward_advection_diverges_when_the_flow_recirculates) {
    // Documents a limitation of the original scheme: the backward difference is
    // anti-diffusive wherever u or v is negative, which the cavity vortex produces.
    Config c = small_cavity(24);
    c.advection = Advection::Backward;
    c.pressure_iterations = 20;
    CpuSolver s(c);
    for (int i = 0; i < 3000; ++i) s.step();
    // Reading the result repairs NaN/Inf, so the divergence shows up in the repair count
    // (or as absurdly large velocities) rather than as non-finite output.
    const Fields& f = s.fields();
    CHECK(s.repaired_values() > 0 || std::fabs(finite_range(f.u).max) > 10.0f * c.u0);
}

TEST(cpu_both_advection_schemes_agree_for_strictly_positive_velocity) {
    // Same plug flow in both schemes: no sign change, so the scheme cannot matter.
    Config a = small_tube(30, 12);
    Config b = a;
    a.advection = Advection::Backward;
    b.advection = Advection::Upwind;
    CpuSolver sa(a), sb(b);
    const std::vector<float> u(a.cells(), a.u0), v(a.cells(), 0.0f);
    sa.set_velocity(u, v);
    sb.set_velocity(u, v);
    sa.step();
    sb.step();
    // v is exactly zero in the interior rows, so the v-upwind branch differs only
    // where wall effects make v tiny and negative; compare u away from the walls.
    for (int y = 2; y <= a.ny - 3; ++y)
        for (int x = 0; x < a.nx; ++x)
            CHECK(sa.fields().u[at(a, x, y)] == sb.fields().u[at(a, x, y)]);
}
