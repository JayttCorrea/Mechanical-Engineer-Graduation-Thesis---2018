#include <cmath>
#include <limits>

#include "tcc/cpu_solver.hpp"
#include "tcc/field_guard.hpp"
#include "test_framework.hpp"

using namespace tcc;

namespace {
const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();
}  // namespace

TEST(guard_leaves_valid_values_alone) {
    FieldGuard g;
    std::vector<float> v = {1.0f, -2.5f, 0.0f, 1e30f};
    const std::vector<float> original = v;
    CHECK(g.repair(v) == 0);
    CHECK(v == original);
}

TEST(guard_replaces_invalid_values_by_the_last_valid_one) {
    FieldGuard g;
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    CHECK(g.repair(v) == 0);  // remembers 1..5

    std::vector<float> next = {10.0f, kNaN, 30.0f, kInf, -kInf};
    CHECK(g.repair(next) == 3);
    CHECK(next[0] == 10.0f);
    CHECK(next[1] == 2.0f);  // last valid value of cell 1
    CHECK(next[2] == 30.0f);
    CHECK(next[3] == 4.0f);
    CHECK(next[4] == 5.0f);
    for (float x : next) CHECK(std::isfinite(x));

    // a cell that stays invalid keeps the value it had before it broke
    std::vector<float> again = {11.0f, kNaN, 31.0f, kNaN, kNaN};
    CHECK(g.repair(again) == 3);
    CHECK(again[1] == 2.0f && again[3] == 4.0f && again[4] == 5.0f);
    CHECK(again[0] == 11.0f);
}

TEST(guard_uses_zero_for_a_cell_that_was_never_valid) {
    FieldGuard g;
    std::vector<float> v = {kNaN, 7.0f, kInf};
    CHECK(g.repair(v) == 2);
    CHECK(v[0] == 0.0f && v[1] == 7.0f && v[2] == 0.0f);
}

TEST(guard_forgets_its_history_when_cleared) {
    FieldGuard g;
    std::vector<float> v = {5.0f, 6.0f};
    g.repair(v);
    g.clear();
    std::vector<float> broken = {kNaN, kNaN};
    CHECK(g.repair(broken) == 2);
    CHECK(broken[0] == 0.0f && broken[1] == 0.0f);
}

TEST(guard_handles_a_changing_size) {
    FieldGuard g;
    std::vector<float> a(10, 1.0f);
    g.repair(a);
    std::vector<float> b(4, kNaN);  // different mesh: history no longer applies
    CHECK(g.repair(b) == 4);
    for (float x : b) CHECK(x == 0.0f);
    std::vector<float> empty;
    CHECK(g.repair(empty) == 0);
}

TEST(cpu_solver_repairs_values_injected_into_the_velocity) {
    Config c = Config::cavity();
    c.nx = c.ny = 20;
    CpuSolver s(c);
    for (int i = 0; i < 30; ++i) s.step();
    const std::vector<float> before_u = s.fields().u;  // a valid read: the guard remembers it
    const std::vector<float> before_v = s.fields().v;
    CHECK(s.repaired_values() == 0);

    std::vector<float> u = before_u, v = before_v;
    u[123] = kNaN;
    v[200] = kInf;
    s.set_velocity(u, v);
    const Fields& f = s.fields();
    CHECK(f.u[123] == before_u[123]);
    CHECK(f.v[200] == before_v[200]);
    CHECK(s.repaired_values() == 2);

    for (int i = 0; i < 20; ++i) s.step();  // the run carries on from valid numbers
    CHECK(all_finite(s.fields().u));
    CHECK(all_finite(s.fields().v));
    CHECK(all_finite(s.fields().p));
    CHECK(all_finite(s.fields().b));
}

TEST(cpu_solver_reports_and_survives_a_real_blow_up) {
    // The thesis convection scheme diverges in the cavity. Reading the result every few
    // steps (as the viewer does every frame) must never show a NaN, and must count repairs.
    Config c = Config::cavity();
    c.nx = c.ny = 24;
    c.advection = Advection::Backward;
    c.pressure_iterations = 20;
    CpuSolver s(c);
    for (int i = 0; i < 4000; ++i) {
        s.step();
        if (i % 20 == 0) {
            const Fields& f = s.fields();
            REQUIRE(all_finite(f.u) && all_finite(f.v) && all_finite(f.p));
        }
    }
    CHECK(s.repaired_values() > 0);
}

TEST(cpu_solver_reset_clears_the_repair_count) {
    Config c = Config::cavity();
    c.nx = c.ny = 12;
    CpuSolver s(c);
    std::vector<float> u(c.cells(), 0.0f), v(c.cells(), 0.0f);
    u[5] = kNaN;
    s.set_velocity(u, v);
    (void)s.fields();
    CHECK(s.repaired_values() == 1);
    s.reset();
    CHECK(s.repaired_values() == 0);
    (void)s.fields();
    CHECK(s.repaired_values() == 0);
}
