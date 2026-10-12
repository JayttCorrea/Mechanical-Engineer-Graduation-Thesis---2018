#include <algorithm>
#include <cmath>

#include "tcc/setup.hpp"
#include "test_framework.hpp"

using namespace tcc;

namespace {

Setup tube(bool gpu = true) { return Setup::defaults(CaseType::Tube, gpu); }
Setup cavity(bool gpu = true) { return Setup::defaults(CaseType::Cavity, gpu); }

}  // namespace

TEST(setup_defaults_reproduce_the_thesis_systems) {
    const Config t = tube().to_config();
    CHECK(t.type == CaseType::Tube);
    CHECK(t.nx == 2000 && t.ny == 100);
    CHECK_NEAR(t.length_x, 2.0, 1e-6);
    CHECK_NEAR(t.length_y, 0.1, 1e-7);
    CHECK_NEAR(t.rho, 999.0, 1e-6);
    CHECK_NEAR(t.nu, 1e-6, 1e-12);
    CHECK_NEAR(t.u0, 0.01, 1e-8);
    CHECK_NEAR(t.dt, 0.01, 1e-9);  // the automatic step lands exactly on the thesis value
    CHECK(t.advection == Advection::Upwind);  // the panel default; Config::tube() keeps the thesis scheme

    const Config c = cavity().to_config();
    CHECK(c.type == CaseType::Cavity);
    CHECK(c.nx == 100 && c.ny == 100);
    CHECK_NEAR(c.dt, 0.01, 1e-9);
    CHECK(c.advection == Advection::Upwind);  // the cavity needs upwind (see the advection tests)
    CHECK(tube().to_config().validate().empty());
    CHECK(c.validate().empty());
}

TEST(setup_backend_follows_gpu_availability) {
    CHECK(tube(true).backend == Backend::Gpu);
    Setup s = tube(false);
    CHECK(s.backend == Backend::Cpu);
    CHECK(!s.editable(Field::Backend));
    s.adjust(Field::Backend, +1);  // there is no GPU to switch to
    CHECK(s.backend == Backend::Cpu);

    Setup g = tube(true);
    g.adjust(Field::Backend, +1);
    CHECK(g.backend == Backend::Cpu);
    g.adjust(Field::Backend, +1);
    CHECK(g.backend == Backend::Gpu);
}

TEST(setup_velocity_steps_through_1_2_5) {
    Setup s = tube();
    s.adjust(Field::Velocity, +1);
    CHECK_NEAR(s.velocity, 0.02, 1e-7);
    s.adjust(Field::Velocity, +1);
    CHECK_NEAR(s.velocity, 0.05, 1e-7);
    s.adjust(Field::Velocity, +1);
    CHECK_NEAR(s.velocity, 0.1, 1e-7);
    s.adjust(Field::Velocity, -1);
    CHECK_NEAR(s.velocity, 0.05, 1e-7);
    for (int i = 0; i < 40; ++i) s.adjust(Field::Velocity, -1);
    CHECK_NEAR(s.velocity, 1e-4, 1e-9);  // lower bound
    for (int i = 0; i < 40; ++i) s.adjust(Field::Velocity, +1);
    CHECK_NEAR(s.velocity, 10.0, 1e-4);  // upper bound
}

TEST(setup_changing_the_system_keeps_fluid_backend_and_sweeps) {
    Setup s = tube();
    s.adjust(Field::Fluid, +1);       // Air
    s.adjust(Field::Sweeps, +1);      // 2
    s.adjust(Field::Backend, +1);     // CPU
    s.adjust(Field::System, +1);
    CHECK(s.type == CaseType::Cavity);
    CHECK(s.nx == 100 && s.ny == 100);
    CHECK_NEAR(s.width, 0.1, 1e-7);
    CHECK(s.fluid == Fluid::Air);
    CHECK(s.sweeps == 2);
    CHECK(s.backend == Backend::Cpu);
    s.adjust(Field::System, +1);
    CHECK(s.type == CaseType::Tube);
    CHECK(s.nx == 2000);
}

TEST(setup_square_cells_derive_the_y_resolution) {
    Setup s = tube();
    CHECK(s.square_cells);
    CHECK(!s.editable(Field::ResolutionY));
    s.adjust(Field::ResolutionX, +1);  // 2010 cells over 2 m -> 0.1 m / (2/2010) = 100.5 -> 101
    CHECK(s.nx == 2010);
    CHECK(s.ny == 101);
    s.adjust(Field::ResolutionY, +1);  // locked: ignored
    CHECK(s.ny == 101);

    s.adjust(Field::SquareCells, +1);  // unlock
    CHECK(s.editable(Field::ResolutionY));
    s.adjust(Field::ResolutionY, +1);
    CHECK(s.ny == 111);
    s.adjust(Field::ResolutionY, +1, /*coarse=*/true);
    CHECK(s.ny == 211);
}

TEST(setup_size_changes_keep_cells_square) {
    Setup s = cavity();
    s.adjust(Field::Height, +1);  // 0.1 -> 0.2 m: twice as tall -> twice the rows
    CHECK(s.ny == 200);
    CHECK(s.nx == 100);
}

TEST(setup_never_exceeds_the_cell_limit) {
    Setup s = tube();
    for (int i = 0; i < 300; ++i) s.adjust(Field::ResolutionX, +1, /*coarse=*/true);
    CHECK(static_cast<long long>(s.nx) * s.ny <= Setup::kMaxCells);
    CHECK(s.nx <= Setup::kMaxSide && s.ny <= Setup::kMaxSide);

    Setup free_form = cavity();
    free_form.adjust(Field::SquareCells, +1);
    for (int i = 0; i < 100; ++i) {
        free_form.adjust(Field::ResolutionX, +1, true);
        free_form.adjust(Field::ResolutionY, +1, true);
    }
    CHECK(static_cast<long long>(free_form.nx) * free_form.ny <= Setup::kMaxCells);

    for (int i = 0; i < 1000; ++i) s.adjust(Field::ResolutionX, -1, true);
    CHECK(s.nx >= Setup::kMinSide && s.ny >= Setup::kMinSide);
}

TEST(setup_fluids_change_the_physics) {
    Setup s = cavity();
    s.adjust(Field::Fluid, +1);
    CHECK(s.fluid == Fluid::Air);
    Config c = s.to_config();
    CHECK_NEAR(c.rho, 1.2, 1e-6);
    CHECK_NEAR(c.nu, 1.5e-5, 1e-10);
    s.adjust(Field::Fluid, +1);
    CHECK(s.fluid == Fluid::Oil);
    s.adjust(Field::Fluid, +1);
    CHECK(s.fluid == Fluid::Water);
    s.adjust(Field::Fluid, -1);
    CHECK(s.fluid == Fluid::Oil);
}

TEST(setup_automatic_time_step_scales_with_the_flow) {
    Setup s = tube();
    CHECK_NEAR(s.dt(), 0.01, 1e-9);
    s.adjust(Field::Velocity, +1);  // 0.02 m/s: CFL limit halves
    CHECK_NEAR(s.dt(), 0.005, 1e-9);
    for (int i = 0; i < 3; ++i) s.adjust(Field::Velocity, +1);  // 0.2 m/s
    CHECK(s.dt() <= 0.001f + 1e-9f);

    Setup viscous = cavity();
    viscous.adjust(Field::Fluid, -1);  // Oil: diffusion limits the step
    CHECK(viscous.dt() < 0.001f);
}

TEST(setup_automatic_time_step_is_always_stable) {
    // Whatever combination the menu produces, the Config must be valid and warning-free.
    Setup s = tube();
    unsigned seed = 12345;
    const Field fields[] = {Field::System, Field::Fluid,       Field::Velocity,   Field::Width,
                            Field::Height, Field::SquareCells, Field::ResolutionX, Field::ResolutionY,
                            Field::Sweeps, Field::Advection};
    for (int i = 0; i < 3000; ++i) {
        seed = seed * 1664525u + 1013904223u;
        const Field f = fields[(seed >> 16) % (sizeof fields / sizeof fields[0])];
        s.adjust(f, ((seed >> 8) & 1) ? +1 : -1, (seed >> 9) & 1);
        const Config c = s.to_config();
        REQUIRE(c.validate().empty());
        CHECK(c.warnings().empty());
        CHECK(c.cfl_number() <= 0.11);
        CHECK(c.diffusion_number() <= 0.11);
    }
}

TEST(setup_manual_time_step_starts_from_the_automatic_one) {
    Setup s = tube();
    s.adjust(Field::Velocity, +1);  // auto dt = 0.005
    s.adjust(Field::TimeStepMode, +1);
    CHECK(!s.auto_dt);
    CHECK_NEAR(s.dt(), 0.005, 1e-9);
    CHECK(s.visible_fields().back() == Field::TimeStep);
    s.adjust(Field::TimeStep, +1);
    CHECK_NEAR(s.dt(), 0.01, 1e-9);
    s.adjust(Field::TimeStepMode, +1);  // back to automatic
    CHECK(s.auto_dt);
    CHECK(s.visible_fields().back() == Field::TimeStepMode);
    s.adjust(Field::TimeStep, +1);      // ignored while automatic
    CHECK_NEAR(s.dt(), 0.005, 1e-9);
}

TEST(setup_round_trips_a_config) {
    for (const Config& original : {Config::tube(), Config::cavity()}) {
        const Setup s = Setup::from_config(original, Backend::Cpu, true);
        const Config back = s.to_config();
        CHECK(back.type == original.type);
        CHECK(back.nx == original.nx && back.ny == original.ny);
        CHECK_NEAR(back.length_x, original.length_x, 1e-6);
        CHECK_NEAR(back.length_y, original.length_y, 1e-7);
        CHECK_NEAR(back.u0, original.u0, 1e-8);
        CHECK_NEAR(back.dt, original.dt, 1e-9);
        CHECK_NEAR(back.rho, original.rho, 1e-6);
        CHECK_NEAR(back.nu, original.nu, 1e-12);
        CHECK(back.advection == original.advection);
        CHECK(back.pressure_iterations == original.pressure_iterations);
        CHECK(s.fluid == Fluid::Water);
        CHECK(s.backend == Backend::Cpu);
    }
}

TEST(setup_keeps_unknown_fluids_from_the_command_line_as_custom) {
    Config c = Config::cavity();
    c.rho = 1234.0f;
    c.nu = 3e-5f;
    Setup s = Setup::from_config(c, Backend::Gpu, true);
    CHECK(s.fluid == Fluid::Custom);
    CHECK_NEAR(s.to_config().rho, 1234.0, 1e-3);
    CHECK_NEAR(s.to_config().nu, 3e-5, 1e-10);
    s.adjust(Field::Fluid, +1);  // leaves Custom for the first preset
    CHECK(s.fluid == Fluid::Water);
}

TEST(setup_warns_about_known_problems) {
    Setup s = cavity();
    CHECK(s.warnings().empty());

    s.adjust(Field::Advection, +1);  // Backward in the cavity
    bool mentions_cavity = false;
    for (const std::string& w : s.warnings()) mentions_cavity = mentions_cavity || w.find("diverges") != std::string::npos;
    CHECK(mentions_cavity);
    s.adjust(Field::Advection, +1);

    for (int i = 0; i < 6; ++i) s.adjust(Field::Velocity, +1);  // 0.01 -> 1 m/s: Re = 1e5
    CHECK(s.reynolds() > 2300.0);
    bool mentions_turbulence = false;
    for (const std::string& w : s.warnings()) mentions_turbulence = mentions_turbulence || w.find("turbulent") != std::string::npos;
    CHECK(mentions_turbulence);

    Setup big = cavity(false);
    for (int i = 0; i < 10; ++i) big.adjust(Field::ResolutionX, +1, true);  // 1100 x 1100 cells
    CHECK(static_cast<long long>(big.nx) * big.ny > 1000000);
    bool mentions_cpu = false;
    for (const std::string& w : big.warnings()) mentions_cpu = mentions_cpu || w.find("CPU") != std::string::npos;
    CHECK(mentions_cpu);
}

TEST(setup_warns_when_the_thesis_scheme_is_used_beyond_its_range) {
    Setup s = tube();
    s.adjust(Field::Advection, +1);  // Backward
    CHECK(s.advection == Advection::Backward);
    CHECK(s.warnings().empty());     // the thesis tube (Re = 1000) ran with it

    s.adjust(Field::Velocity, +1);   // twice as fast as the thesis
    bool warned = false;
    for (const std::string& w : s.warnings()) warned = warned || w.find("blows up") != std::string::npos;
    CHECK(warned);

    s.adjust(Field::Advection, +1);  // Upwind
    for (const std::string& w : s.warnings()) CHECK(w.find("Backward") == std::string::npos);
}

TEST(setup_texts_are_never_empty) {
    for (Setup s : {tube(), cavity()}) {
        s.adjust(Field::TimeStepMode, +1);  // make the TimeStep row visible too
        for (Field f : s.visible_fields()) {
            CHECK(!s.label(f).empty());
            CHECK(!s.value_text(f).empty());
        }
        CHECK(!s.summary().empty());
    }
    CHECK(tube().summary().find("200,000 cells") != std::string::npos);
    CHECK(tube().label(Field::Velocity).find("Inlet") != std::string::npos);
    CHECK(cavity().label(Field::Velocity).find("Lid") != std::string::npos);
}

// ---- typed input -------------------------------------------------------------------

TEST(setup_typed_values_are_accepted_and_applied) {
    Setup s = tube();
    CHECK(s.set_value(Field::Velocity, "0.05") == SetResult::Ok);
    CHECK_NEAR(s.velocity, 0.05, 1e-7);
    CHECK(s.set_value(Field::Width, "3.5") == SetResult::Ok);
    CHECK_NEAR(s.width, 3.5, 1e-6);
    CHECK(s.set_value(Field::Height, "  0.2 ") == SetResult::Ok);  // spaces are ignored
    CHECK_NEAR(s.height, 0.2, 1e-7);
    CHECK(s.set_value(Field::ResolutionX, "800") == SetResult::Ok);
    CHECK(s.nx == 800);
    CHECK(s.ny == 46);  // square cells: 800 * 0.2 / 3.5 = 45.7
    CHECK(s.set_value(Field::Sweeps, "30") == SetResult::Ok);
    CHECK(s.sweeps == 30);
}

TEST(setup_typed_values_accept_comma_and_exponent) {
    Setup s = cavity();
    CHECK(s.set_value(Field::Velocity, "0,25") == SetResult::Ok);  // decimal comma
    CHECK_NEAR(s.velocity, 0.25, 1e-7);
    CHECK(s.set_value(Field::Velocity, "5e-3") == SetResult::Ok);
    CHECK_NEAR(s.velocity, 0.005, 1e-8);
    CHECK(s.set_value(Field::Velocity, "1E-2") == SetResult::Ok);
    CHECK_NEAR(s.velocity, 0.01, 1e-8);
}

TEST(setup_typed_values_outside_the_range_are_limited) {
    Setup s = tube();
    CHECK(s.set_value(Field::Velocity, "500") == SetResult::Limited);
    CHECK_NEAR(s.velocity, 10.0, 1e-5);
    CHECK(s.set_value(Field::Velocity, "0") == SetResult::Limited);
    CHECK_NEAR(s.velocity, 1e-4, 1e-9);
    CHECK(s.set_value(Field::Velocity, "-3") == SetResult::Limited);
    CHECK_NEAR(s.velocity, 1e-4, 1e-9);
    CHECK(s.set_value(Field::ResolutionX, "999999") == SetResult::Limited);
    CHECK(s.nx <= Setup::kMaxSide);
    CHECK(static_cast<long long>(s.nx) * s.ny <= Setup::kMaxCells);
    CHECK(s.set_value(Field::ResolutionX, "3") == SetResult::Limited);
    CHECK(s.nx == Setup::kMinSide);
    CHECK(s.set_value(Field::Sweeps, "5000") == SetResult::Limited);
    CHECK(s.sweeps == 1000);
}

TEST(setup_rejects_text_that_is_not_a_number) {
    const Setup before = tube();
    Setup t = tube();
    for (const char* bad : {"", "   ", "abc", "1.2.3", "5x", "--4", "nan", "inf", "0x10", "1e"}) {
        CHECK(t.set_value(Field::Velocity, bad) == SetResult::Invalid);
    }
    CHECK(t.same_simulation(before));  // invalid input changed nothing

    // a number too large for a float is understood and limited, never turned into inf/NaN
    CHECK(t.set_value(Field::Velocity, "1e999999") == SetResult::Invalid);
    CHECK(t.set_value(Field::Velocity, "1e300") == SetResult::Limited);
    CHECK(std::isfinite(t.velocity));
}

TEST(setup_only_some_rows_take_typed_values) {
    Setup s = tube();
    CHECK(s.is_numeric(Field::Velocity));
    CHECK(s.is_numeric(Field::ResolutionX));
    CHECK(!s.is_numeric(Field::System));
    CHECK(!s.is_numeric(Field::Advection));
    CHECK(!s.is_numeric(Field::ResolutionY));  // derived while cells are square
    CHECK(s.set_value(Field::ResolutionY, "50") == SetResult::Invalid);
    CHECK(!s.is_numeric(Field::TimeStep));     // automatic time step
    s.adjust(Field::SquareCells, +1);
    CHECK(s.is_numeric(Field::ResolutionY));
    CHECK(s.set_value(Field::ResolutionY, "50") == SetResult::Ok);
    CHECK(s.ny == 50);
    s.adjust(Field::TimeStepMode, +1);
    CHECK(s.is_numeric(Field::TimeStep));
    CHECK(s.set_value(Field::TimeStep, "0.002") == SetResult::Ok);
    CHECK_NEAR(s.dt(), 0.002, 1e-9);
    CHECK(s.set_value(Field::System, "1") == SetResult::Invalid);
}

TEST(setup_edit_text_round_trips) {
    Setup s = cavity();
    s.adjust(Field::SquareCells, +1);
    s.adjust(Field::TimeStepMode, +1);
    for (Field f : {Field::Velocity, Field::Width, Field::Height, Field::ResolutionX, Field::ResolutionY,
                    Field::Sweeps, Field::TimeStep}) {
        const Setup before = s;
        CHECK(!s.edit_text(f).empty());
        CHECK(!s.range_text(f).empty());
        CHECK(s.set_value(f, s.edit_text(f)) == SetResult::Ok);  // re-entering a value changes nothing
        CHECK(s.same_simulation(before));
    }
}

TEST(setup_random_typed_input_always_leaves_a_valid_setup) {
    Setup s = tube();
    s.adjust(Field::SquareCells, +1);
    s.adjust(Field::TimeStepMode, +1);
    const char* inputs[] = {"0.5", "1e9", "-1", "0", "7", "123456", "1e-9", "abc", "2,5", "4000", "9", "0.0001"};
    const Field fields[] = {Field::Velocity, Field::Width, Field::Height, Field::ResolutionX,
                            Field::ResolutionY, Field::Sweeps, Field::TimeStep};
    unsigned seed = 99;
    for (int i = 0; i < 3000; ++i) {
        seed = seed * 1664525u + 1013904223u;
        s.set_value(fields[(seed >> 16) % 7], inputs[(seed >> 8) % 12]);
        const Config c = s.to_config();
        REQUIRE(c.validate().empty());
        CHECK(static_cast<long long>(c.nx) * c.ny <= Setup::kMaxCells);
    }
}

TEST(setup_same_simulation_detects_changes) {
    const Setup a = tube();
    CHECK(a.same_simulation(tube()));
    Setup b = a;
    b.adjust(Field::Velocity, +1);
    CHECK(!a.same_simulation(b));
    b = a;
    b.adjust(Field::Sweeps, +1);
    CHECK(!a.same_simulation(b));
    b = a;
    b.adjust(Field::Backend, +1);
    CHECK(!a.same_simulation(b));
    b = a;
    b.adjust(Field::Fluid, +1);
    CHECK(!a.same_simulation(b));
}
