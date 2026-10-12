// GPU (OpenCL) solver tests. They are skipped (exit code 77) when no device exists.

#include <algorithm>
#include <cmath>
#include <limits>

#include "tcc/cpu_solver.hpp"
#include "tcc/gpu_solver.hpp"
#include "tcc/stabilizer.hpp"
#include "test_framework.hpp"

using namespace tcc;

namespace {

void require_device() {
    if (list_opencl_devices().empty()) SKIP_TEST("no OpenCL device");
}

double max_abs(const std::vector<float>& a) {
    double m = 0.0;
    for (float x : a) m = std::max(m, std::fabs(static_cast<double>(x)));
    return m;
}

// Largest |a - b|, divided by `scale`. Velocity components share one scale (the fastest
// component), otherwise a nearly-zero v would make float rounding look like a big error.
double scaled_difference(const std::vector<float>& a, const std::vector<float>& b, double scale) {
    if (a.size() != b.size()) return 1e30;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
    return worst / std::max(scale, 1e-30);
}

double relative_difference(const std::vector<float>& a, const std::vector<float>& b) {
    return scaled_difference(a, b, max_abs(b));
}

void compare_cpu_gpu(Config c, int steps, int ipf) {
    c.pressure_iterations = ipf;
    CpuSolver cpu(c);
    GpuSolver gpu(c);
    for (int i = 0; i < steps; ++i) {
        cpu.step();
        gpu.step();
    }
    gpu.wait();
    const Fields& a = gpu.fields();
    const Fields& b = cpu.fields();
    REQUIRE(max_abs(b.u) > 0.0);
    const double velocity_scale = std::max(max_abs(b.u), max_abs(b.v));
    CHECK(scaled_difference(a.u, b.u, velocity_scale) < 1e-5);
    CHECK(scaled_difference(a.v, b.v, velocity_scale) < 1e-5);
    CHECK(relative_difference(a.p, b.p) < 1e-4);
    CHECK(relative_difference(a.b, b.b) < 1e-4);
}

}  // namespace

TEST(gpu_lists_at_least_one_device) {
    require_device();
    const auto devices = list_opencl_devices();
    CHECK(!devices.empty());
    CHECK(!devices.front().name.empty());
}

TEST(gpu_matches_cpu_on_the_tube) {
    require_device();
    Config c = Config::tube();
    c.nx = 64;
    c.ny = 20;
    compare_cpu_gpu(c, 40, 1);
    compare_cpu_gpu(c, 25, 8);
}

TEST(gpu_matches_cpu_on_the_cavity) {
    require_device();
    Config c = Config::cavity();
    c.nx = c.ny = 40;
    compare_cpu_gpu(c, 40, 1);
    compare_cpu_gpu(c, 25, 8);
}

TEST(gpu_matches_cpu_on_an_awkward_mesh) {
    require_device();  // sizes that are not multiples of any work-group size
    Config c = Config::tube();
    c.nx = 37;
    c.ny = 11;
    compare_cpu_gpu(c, 30, 3);
    Config k = Config::cavity();
    k.nx = 23;
    k.ny = 31;
    compare_cpu_gpu(k, 30, 3);
}

TEST(gpu_matches_cpu_with_warm_start_pressure) {
    require_device();
    Config c = Config::cavity();
    c.nx = c.ny = 24;
    c.warm_start_pressure = true;
    compare_cpu_gpu(c, 40, 4);
}

TEST(gpu_set_velocity_round_trips_and_validates) {
    require_device();
    Config c = Config::cavity();
    c.nx = 12;
    c.ny = 9;
    GpuSolver gpu(c);
    std::vector<float> u(c.cells()), v(c.cells());
    for (std::size_t i = 0; i < u.size(); ++i) {
        u[i] = static_cast<float>(i) * 1e-3f;
        v[i] = -static_cast<float>(i) * 2e-3f;
    }
    gpu.set_velocity(u, v);
    CHECK(gpu.fields().u == u);
    CHECK(gpu.fields().v == v);
    CHECK_THROWS(gpu.set_velocity(std::vector<float>(3), v));
}

TEST(gpu_reset_returns_to_rest_and_replays_identically) {
    require_device();
    Config c = Config::tube();
    c.nx = 30;
    c.ny = 12;
    GpuSolver gpu(c);
    for (int i = 0; i < 20; ++i) gpu.step();
    const std::vector<float> first = gpu.fields().u;
    CHECK(max_abs(first) > 0.0);

    gpu.reset();
    CHECK(gpu.steps() == 0);
    CHECK(max_abs(gpu.fields().u) == 0.0);
    for (int i = 0; i < 20; ++i) gpu.step();
    CHECK(gpu.fields().u == first);  // deterministic on the same device
}

TEST(gpu_changing_sweeps_between_steps_matches_cpu) {
    require_device();
    Config c = Config::cavity();
    c.nx = c.ny = 20;
    CpuSolver cpu(c);
    GpuSolver gpu(c);
    for (int i = 0; i < 30; ++i) {
        const int ipf = 1 + (i % 4) * 3;
        cpu.set_pressure_iterations(ipf);
        gpu.set_pressure_iterations(ipf);
        cpu.step();
        gpu.step();
    }
    CHECK(relative_difference(gpu.fields().u, cpu.fields().u) < 1e-4);
}

TEST(gpu_repairs_values_that_come_back_invalid) {
    require_device();
    Config c = Config::cavity();
    c.nx = c.ny = 20;
    GpuSolver gpu(c);
    for (int i = 0; i < 30; ++i) gpu.step();
    const std::vector<float> before_u = gpu.fields().u;  // a valid read
    const std::vector<float> before_v = gpu.fields().v;
    CHECK(gpu.repaired_values() == 0);

    // poison the device with NaN / Inf, then read it back
    std::vector<float> u = before_u, v = before_v;
    u[77] = std::numeric_limits<float>::quiet_NaN();
    v[310] = std::numeric_limits<float>::infinity();
    gpu.set_velocity(u, v);
    const Fields& f = gpu.fields();
    CHECK(f.u[77] == before_u[77]);  // the last valid value of that cell
    CHECK(f.v[310] == before_v[310]);
    CHECK(gpu.repaired_values() == 2);

    // the corrected values were written back: the device state is valid again
    for (int i = 0; i < 20; ++i) gpu.step();
    CHECK(all_finite(gpu.fields().u));
    CHECK(all_finite(gpu.fields().v));
    CHECK(all_finite(gpu.fields().p));
    CHECK(all_finite(gpu.fields().b));
}

TEST(gpu_survives_a_real_blow_up_without_returning_nan) {
    require_device();
    // Thesis convection scheme in the cavity: it diverges. Reading every few steps, as the
    // viewer does, must always give finite fields and report the repairs.
    Config c = Config::cavity();
    c.nx = c.ny = 40;
    c.advection = Advection::Backward;
    c.pressure_iterations = 20;
    GpuSolver gpu(c);
    for (int i = 0; i < 6000; ++i) {
        gpu.step();
        if (i % 25 == 0) {
            const Fields& f = gpu.fields();
            REQUIRE(all_finite(f.u) && all_finite(f.v) && all_finite(f.p));
        }
    }
    CHECK(gpu.repaired_values() > 0);

    gpu.reset();
    CHECK(gpu.repaired_values() == 0);
}

TEST(gpu_follows_time_step_and_scheme_changes_made_while_running) {
    require_device();
    Config c = Config::cavity();
    c.nx = c.ny = 24;
    c.advection = Advection::Backward;
    CpuSolver cpu(c);
    GpuSolver gpu(c);
    for (int i = 0; i < 20; ++i) { cpu.step(); gpu.step(); }
    for (Solver* s : {static_cast<Solver*>(&cpu), static_cast<Solver*>(&gpu)}) {
        s->set_time_step(c.dt / 2.0f);
        s->set_advection(Advection::Upwind);
    }
    for (int i = 0; i < 40; ++i) { cpu.step(); gpu.step(); }
    const double scale = std::max(max_abs(cpu.fields().u), max_abs(cpu.fields().v));
    CHECK(scaled_difference(gpu.fields().u, cpu.fields().u, scale) < 1e-5);
    CHECK(scaled_difference(gpu.fields().v, cpu.fields().v, scale) < 1e-5);
}

TEST(gpu_velocity_limit_repairs_huge_but_finite_values) {
    require_device();
    Config c = Config::cavity();
    c.nx = c.ny = 16;
    GpuSolver gpu(c);
    gpu.set_velocity_limit(10.0f * c.u0);
    std::vector<float> u(c.cells(), 0.0f), v(c.cells(), 0.0f);
    u[40] = 5.0f;  // finite, but 500x the lid speed
    gpu.set_velocity(u, v);
    CHECK(gpu.fields().u[40] == 0.0f);
    CHECK(gpu.repaired_values() == 1);
}

TEST(gpu_stabilizer_keeps_a_diverging_cavity_running) {
    require_device();
    Config c = Config::cavity();
    c.nx = c.ny = 40;
    c.advection = Advection::Backward;
    c.pressure_iterations = 20;
    GpuSolver gpu(c);
    gpu.set_velocity_limit(10.0f * c.u0);
    Stabilizer stabilizer(c.dt, c.advection);

    auto run_frames = [&](int frames) {
        for (int f = 0; f < frames; ++f) {
            for (int i = 0; i < 25; ++i) gpu.step();
            const Fields& fields = gpu.fields();
            REQUIRE(all_finite(fields.u) && all_finite(fields.v) && all_finite(fields.p));
            if (stabilizer.update(gpu.repaired_values(), gpu.steps())) {
                gpu.set_time_step(stabilizer.dt());
                gpu.set_advection(stabilizer.advection());
            }
        }
    };
    run_frames(200);  // 5000 steps: instability, repairs, stabiliser reacts
    CHECK(gpu.repaired_values() > 0);
    CHECK(stabilizer.advection() == Advection::Upwind);

    const std::uint64_t repaired_before = gpu.repaired_values();
    const std::vector<float> before = gpu.fields().u;
    run_frames(120);  // 3000 more steps: quiet and still evolving
    CHECK(gpu.repaired_values() == repaired_before);
    CHECK(gpu.fields().u != before);
}

TEST(gpu_rejects_invalid_configuration_and_bad_device_index) {
    require_device();
    Config c = Config::cavity();
    c.nx = 2;
    CHECK_THROWS(GpuSolver{c});
    GpuOptions o;
    o.device_index = 999;
    CHECK_THROWS(GpuSolver(Config::cavity(), o));
}

TEST(gpu_matches_cpu_with_both_advection_schemes) {
    require_device();
    for (Advection scheme : {Advection::Backward, Advection::Upwind}) {
        Config c = Config::cavity();
        c.nx = c.ny = 32;
        c.advection = scheme;
        compare_cpu_gpu(c, 60, 4);
        Config t = Config::tube();
        t.nx = 48;
        t.ny = 16;
        t.advection = scheme;
        compare_cpu_gpu(t, 60, 4);
    }
}
