#include "tcc/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace tcc {

const char* to_string(CaseType type) {
    switch (type) {
        case CaseType::Tube: return "tube";
        case CaseType::Cavity: return "cavity";
    }
    return "?";
}

bool parse_case(const std::string& text, CaseType& out) {
    if (text == "tube") { out = CaseType::Tube; return true; }
    if (text == "cavity") { out = CaseType::Cavity; return true; }
    return false;
}

const char* to_string(Advection scheme) {
    return scheme == Advection::Backward ? "backward" : "upwind";
}

bool parse_advection(const std::string& text, Advection& out) {
    if (text == "backward") { out = Advection::Backward; return true; }
    if (text == "upwind") { out = Advection::Upwind; return true; }
    return false;
}

double Config::cfl_number() const {
    const double speed = std::max(std::fabs(static_cast<double>(u0)), std::fabs(static_cast<double>(v0)));
    const double h = std::min(static_cast<double>(dx()), static_cast<double>(dy()));
    return speed * static_cast<double>(dt) / h;
}

double Config::diffusion_number() const {
    const double ddx = static_cast<double>(dx());
    const double ddy = static_cast<double>(dy());
    return static_cast<double>(nu) * static_cast<double>(dt) * (1.0 / (ddx * ddx) + 1.0 / (ddy * ddy));
}

std::string Config::validate() const {
    if (nx < 3 || ny < 3) return "the mesh needs at least 3x3 cells";
    // Cell indices are 32-bit ints in the kernels.
    if (cells() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
        return "the mesh has too many cells";
    if (!(length_x > 0.0f) || !(length_y > 0.0f) || !std::isfinite(length_x) || !std::isfinite(length_y))
        return "domain lengths must be positive and finite";
    if (!(rho > 0.0f) || !std::isfinite(rho)) return "density must be positive and finite";
    if (!(nu >= 0.0f) || !std::isfinite(nu)) return "viscosity must be >= 0 and finite";
    if (!(dt > 0.0f) || !std::isfinite(dt)) return "time step must be positive and finite";
    if (!std::isfinite(u0) || !std::isfinite(v0)) return "inlet/lid velocity must be finite";
    if (pressure_iterations < 1) return "pressure_iterations must be >= 1";
    return {};
}

std::vector<std::string> Config::warnings() const {
    std::vector<std::string> out;
    char buf[160];
    if (cfl_number() > 0.5) {
        std::snprintf(buf, sizeof buf,
                      "advection CFL number is %.3f (> 0.5): the explicit scheme may be unstable, reduce dt",
                      cfl_number());
        out.emplace_back(buf);
    }
    if (diffusion_number() > 0.25) {
        std::snprintf(buf, sizeof buf,
                      "diffusion number is %.3f (> 0.25): the explicit scheme may be unstable, reduce dt",
                      diffusion_number());
        out.emplace_back(buf);
    }
    return out;
}

Config Config::tube() {
    return Config{};  // defaults are the thesis tube
}

Config Config::cavity() {
    Config c;
    c.type = CaseType::Cavity;
    c.nx = 100;
    c.ny = 100;
    c.length_x = 0.1f;
    c.length_y = 0.1f;
    c.advection = Advection::Upwind;  // the lid-driven flow recirculates (negative velocities)
    return c;
}

}  // namespace tcc
