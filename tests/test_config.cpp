#include <limits>

#include "tcc/config.hpp"
#include "tcc/cpu_solver.hpp"
#include "test_framework.hpp"

using namespace tcc;

TEST(config_presets_match_the_thesis) {
    const Config t = Config::tube();
    CHECK(t.type == CaseType::Tube);
    CHECK(t.nx == 2000 && t.ny == 100);
    // float members vs. double literals: compare to float precision
    CHECK_NEAR(t.length_x, 2.0, 1e-6);
    CHECK_NEAR(t.length_y, 0.1, 1e-7);
    CHECK_NEAR(t.rho, 999.0, 1e-6);
    CHECK_NEAR(t.dt, 0.01, 1e-8);
    CHECK_NEAR(t.u0, 0.01, 1e-8);
    CHECK_NEAR(t.v0, 0.0, 1e-12);
    CHECK_NEAR(t.nu, 1e-6, 1e-12);
    CHECK(t.pressure_iterations == 1);
    CHECK(!t.warm_start_pressure);

    const Config c = Config::cavity();
    CHECK(c.type == CaseType::Cavity);
    CHECK(c.nx == c.ny);
    CHECK_NEAR(c.length_x, 0.1, 1e-7);
    CHECK_NEAR(c.length_y, 0.1, 1e-7);
}

TEST(config_presets_are_valid_and_stable) {
    CHECK(Config::tube().validate().empty());
    CHECK(Config::cavity().validate().empty());
    CHECK(Config::tube().warnings().empty());
    CHECK(Config::cavity().warnings().empty());
}

TEST(config_cell_sizes) {
    Config c = Config::tube();
    CHECK_NEAR(c.dx(), 0.001, 1e-9);
    CHECK_NEAR(c.dy(), 0.001, 1e-9);
    CHECK(c.cells() == 200000);
}

TEST(config_rejects_bad_values) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto bad = [](Config c) { return !c.validate().empty(); };
    Config c = Config::cavity();

    Config x = c; x.nx = 2;            CHECK(bad(x));
    x = c;        x.ny = 0;            CHECK(bad(x));
    x = c;        x.nx = 100000; x.ny = 100000; CHECK(bad(x));  // too many cells for 32-bit indices
    x = c;        x.length_x = 0.0f;   CHECK(bad(x));
    x = c;        x.length_y = -1.0f;  CHECK(bad(x));
    x = c;        x.rho = 0.0f;        CHECK(bad(x));
    x = c;        x.nu = -1e-6f;       CHECK(bad(x));
    x = c;        x.dt = 0.0f;         CHECK(bad(x));
    x = c;        x.dt = nan;          CHECK(bad(x));
    x = c;        x.u0 = nan;          CHECK(bad(x));
    x = c;        x.pressure_iterations = 0; CHECK(bad(x));
}

TEST(config_warns_about_unstable_time_steps) {
    Config c = Config::cavity();
    c.dt = 1.0f;  // CFL = 0.01 * 1 / 0.001 = 10
    CHECK(c.cfl_number() > 0.5);
    CHECK(!c.warnings().empty());

    c = Config::cavity();
    c.nu = 1e-3f;  // diffusion number far above 0.25
    CHECK(c.diffusion_number() > 0.25);
    CHECK(!c.warnings().empty());
}

TEST(config_case_name_parsing) {
    CaseType t = CaseType::Cavity;
    CHECK(parse_case("tube", t) && t == CaseType::Tube);
    CHECK(parse_case("cavity", t) && t == CaseType::Cavity);
    CHECK(!parse_case("pipe", t));
    CHECK(std::string(to_string(CaseType::Tube)) == "tube");
}

TEST(solver_constructor_validates_the_config) {
    Config c = Config::cavity();
    c.nx = 1;
    CHECK_THROWS(CpuSolver{c});
}
