#include <cmath>
#include <limits>

#include "tcc/cpu_solver.hpp"
#include "tcc/field_guard.hpp"
#include "tcc/stabilizer.hpp"
#include "test_framework.hpp"

using namespace tcc;

namespace {

// What the viewer does after every displayed frame: run some steps, read the fields (which
// repairs invalid values), then let the stabiliser decide and apply its answer.
struct FrameStats {
    std::uint64_t events = 0;
    std::uint64_t repaired_in_last_window = 0;
};

void drive(Solver& solver, Stabilizer& stabilizer, int total_steps, int steps_per_frame, int tail_window,
           FrameStats& stats) {
    std::uint64_t repaired_at_window_start = solver.repaired_values();  // window = tail of this call
    for (int done = 0; done < total_steps; done += steps_per_frame) {
        for (int i = 0; i < steps_per_frame; ++i) solver.step();
        (void)solver.fields();
        if (stabilizer.update(solver.repaired_values(), solver.steps())) {
            solver.set_time_step(stabilizer.dt());
            solver.set_advection(stabilizer.advection());
        }
        if (done + steps_per_frame <= total_steps - tail_window) repaired_at_window_start = solver.repaired_values();
    }
    stats.events = stabilizer.events();
    stats.repaired_in_last_window = solver.repaired_values() - repaired_at_window_start;
}

}  // namespace

// ---- pure logic ------------------------------------------------------------------

TEST(stabilizer_starts_with_the_base_settings) {
    Stabilizer s(0.01f, Advection::Backward);
    CHECK_NEAR(s.dt(), 0.01, 1e-9);
    CHECK(s.advection() == Advection::Backward);
    CHECK(s.level() == 0);
    CHECK(!s.update(0, 0));
    CHECK(!s.update(0, 1000));
    CHECK(s.events() == 0);
}

TEST(stabilizer_replaces_the_thesis_convection_before_touching_the_time_step) {
    Stabilizer s(0.01f, Advection::Backward);
    CHECK(s.update(500, 100));  // first event
    CHECK(s.advection() == Advection::Upwind);
    CHECK(s.advection_was_changed());
    CHECK(s.level() == 0);
    CHECK_NEAR(s.dt(), 0.01, 1e-9);

    CHECK(s.update(900, 200));  // still invalid values: now the time step
    CHECK(s.level() == 1);
    CHECK_NEAR(s.dt(), 0.005, 1e-9);
    CHECK(s.events() == 2);
}

TEST(stabilizer_halves_the_time_step_with_upwind_convection) {
    Stabilizer s(0.01f, Advection::Upwind);
    for (int k = 1; k <= 3; ++k) {
        CHECK(s.update(static_cast<std::uint64_t>(k) * 10, static_cast<std::uint64_t>(k) * 100));
        CHECK(s.level() == k);
    }
    CHECK_NEAR(s.dt(), 0.01 / 8.0, 1e-9);
    CHECK(!s.advection_was_changed());
}

TEST(stabilizer_stops_at_the_smallest_time_step) {
    Stabilizer s(1.0f, Advection::Upwind);
    std::uint64_t repaired = 0, steps = 0;
    for (int k = 0; k < Stabilizer::kMaxLevel; ++k) {
        repaired += 5;
        steps += 10;
        CHECK(s.update(repaired, steps));
    }
    CHECK(s.level() == Stabilizer::kMaxLevel);
    CHECK(!s.update(repaired + 5, steps + 10));  // nothing left to try
    CHECK(s.level() == Stabilizer::kMaxLevel);
    CHECK_NEAR(s.dt(), std::ldexp(1.0, -Stabilizer::kMaxLevel), 1e-12);
}

TEST(stabilizer_gives_the_time_step_back_after_a_quiet_period) {
    Stabilizer s(0.01f, Advection::Upwind);
    CHECK(s.update(10, 100));
    CHECK(s.update(20, 200));
    CHECK(s.level() == 2);

    CHECK(!s.update(20, 200 + Stabilizer::kRecoverySteps / 2));  // not quiet for long enough
    CHECK(s.update(20, 200 + Stabilizer::kRecoverySteps));       // one level back
    CHECK(s.level() == 1);
    CHECK(s.update(20, 200 + 2 * Stabilizer::kRecoverySteps));
    CHECK(s.level() == 0);
    CHECK_NEAR(s.dt(), 0.01, 1e-9);
    CHECK(!s.update(20, 200 + 5 * Stabilizer::kRecoverySteps));  // nothing more to give back
}

TEST(stabilizer_a_new_event_restarts_the_quiet_period) {
    Stabilizer s(0.01f, Advection::Upwind);
    CHECK(s.update(10, 100));
    CHECK(!s.update(10, 100 + Stabilizer::kRecoverySteps - 10));
    CHECK(s.update(11, 100 + Stabilizer::kRecoverySteps));  // an event, not a recovery
    CHECK(s.level() == 2);
    CHECK(!s.update(11, 100 + Stabilizer::kRecoverySteps + 100));
}

TEST(stabilizer_survives_a_solver_reset) {
    Stabilizer s(0.01f, Advection::Upwind);
    CHECK(s.update(1000, 5000));
    CHECK(s.level() == 1);
    // the solver was reset: both counters restart from zero, which is not an event
    CHECK(!s.update(0, 0));
    CHECK(!s.update(0, 50));
    CHECK(s.level() == 1);
    CHECK(s.update(7, 100));  // but new repairs after the reset are
    CHECK(s.level() == 2);
}

TEST(stabilizer_reset_restores_the_base_settings) {
    Stabilizer s(0.02f, Advection::Backward);
    s.update(10, 100);
    s.update(20, 200);
    s.reset();
    CHECK(s.level() == 0);
    CHECK(s.advection() == Advection::Backward);
    CHECK_NEAR(s.dt(), 0.02, 1e-9);
    CHECK(s.events() == 0);
}

// ---- guard limit ---------------------------------------------------------------------

TEST(guard_limit_treats_huge_finite_values_as_invalid) {
    FieldGuard g;
    g.set_limit(0.1f);
    std::vector<float> v = {0.05f, -0.05f, 0.0f};
    CHECK(g.repair(v) == 0);
    std::vector<float> next = {0.5f, -1e30f, 0.09f};
    CHECK(g.repair(next) == 2);
    CHECK(next[0] == 0.05f && next[1] == -0.05f && next[2] == 0.09f);

    g.set_limit(0.0f);  // 0 / negative disables the limit again
    std::vector<float> big = {1e20f, 5.0f, 2.0f};
    CHECK(g.repair(big) == 0);
}

TEST(solver_velocity_limit_is_opt_in) {
    Config c = Config::cavity();
    c.nx = c.ny = 10;
    CpuSolver s(c);
    std::vector<float> u(c.cells(), 0.5f), v(c.cells(), 0.0f);  // 50x the lid speed
    s.set_velocity(u, v);
    CHECK(s.fields().u == u);  // no limit by default: only NaN / infinity are repaired
    CHECK(s.repaired_values() == 0);

    s.set_velocity_limit(10.0f * c.u0);
    s.set_velocity(u, v);
    (void)s.fields();
    CHECK(s.repaired_values() == c.cells());
    for (float x : s.fields().u) CHECK(x == 0.5f);  // "last valid" is the previous read, which was 0.5
}

TEST(solver_changes_time_step_and_scheme_while_running) {
    Config c = Config::cavity();
    c.nx = c.ny = 12;
    CpuSolver s(c);
    s.set_time_step(0.002f);
    CHECK_NEAR(s.config().dt, 0.002, 1e-9);
    s.set_advection(Advection::Backward);
    CHECK(s.config().advection == Advection::Backward);
    CHECK_THROWS(s.set_time_step(0.0f));
    CHECK_THROWS(s.set_time_step(-1.0f));
    CHECK_THROWS(s.set_time_step(std::numeric_limits<float>::quiet_NaN()));
    CHECK_NEAR(s.config().dt, 0.002, 1e-9);  // rejected values change nothing
}

// ---- the whole loop on a real solver ------------------------------------------------------

TEST(stabilized_thesis_scheme_keeps_the_cavity_running) {
    // The thesis convection diverges in the cavity. With the stabiliser the run must switch
    // to upwind, stop needing repairs, and keep evolving instead of freezing.
    Config c = Config::cavity();
    c.nx = c.ny = 24;
    c.advection = Advection::Backward;
    c.pressure_iterations = 20;
    CpuSolver s(c);
    s.set_velocity_limit(10.0f * c.u0);
    Stabilizer st(c.dt, c.advection);

    FrameStats stats;
    drive(s, st, 3000, 20, 1000, stats);
    const std::vector<float> at_3000 = s.fields().u;
    drive(s, st, 1000, 20, 1000, stats);
    const std::vector<float> at_4000 = s.fields().u;

    CHECK(st.advection() == Advection::Upwind);
    CHECK(stats.events > 0);
    CHECK(s.repaired_values() > 0);
    CHECK(stats.repaired_in_last_window == 0);  // the instability is gone
    CHECK(all_finite(at_4000));
    for (float x : at_4000) CHECK(std::fabs(x) <= 10.0f * c.u0);
    CHECK(at_3000 != at_4000);                  // and the flow is still evolving
    // the lid still drags the fluid
    CHECK(s.fields().u[static_cast<std::size_t>(c.ny - 1) * static_cast<std::size_t>(c.nx) +
                       static_cast<std::size_t>(c.nx / 2)] > 0.0f);
}

TEST(stabilized_run_recovers_from_a_time_step_that_is_far_too_large) {
    // Tube 0.2 m long, 60 x 20 cells, 0.1 m/s inflow: the CFL number of dt = 0.1 s is 3.
    Config c = Config::tube();
    c.nx = 60;
    c.ny = 20;
    c.length_x = 0.2f;
    c.u0 = 0.1f;
    c.dt = 0.1f;
    c.advection = Advection::Upwind;
    CpuSolver s(c);
    s.set_velocity_limit(10.0f * c.u0);
    Stabilizer st(c.dt, c.advection);

    FrameStats stats;
    drive(s, st, 4000, 20, 1000, stats);

    CHECK(st.level() > 0);
    CHECK(st.dt() < c.dt);
    CHECK(st.events() <= static_cast<std::uint64_t>(Stabilizer::kMaxLevel) + 2);  // converges, no endless cycle
    CHECK(stats.repaired_in_last_window == 0);
    CHECK(all_finite(s.fields().u) && all_finite(s.fields().v));
    // the Courant number of the final time step is back in the stable range
    Config now = s.config();
    CHECK(now.cfl_number() < 1.0);
    // fluid has entered the tube from the left
    CHECK(s.fields().u[static_cast<std::size_t>(c.ny / 2) * static_cast<std::size_t>(c.nx) + 1] > 0.5f * c.u0);
}

TEST(a_stable_run_is_left_completely_alone) {
    Config c = Config::cavity();
    c.nx = c.ny = 20;
    CpuSolver s(c);
    s.set_velocity_limit(10.0f * c.u0);
    Stabilizer st(c.dt, c.advection);
    FrameStats stats;
    drive(s, st, 2000, 20, 500, stats);
    CHECK(stats.events == 0);
    CHECK(st.level() == 0);
    CHECK(s.repaired_values() == 0);
    CHECK_NEAR(s.config().dt, c.dt, 1e-9);
}
