#include "tcc/setup.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tcc {

namespace {

constexpr FluidProperties kPresets[] = {
    {"Water", 999.0f, 1.0e-6f},
    {"Air", 1.2f, 1.5e-5f},
    {"Oil", 900.0f, 1.0e-4f},
};

// Next value in the 1-2-5 sequence (..., 0.5, 1, 2, 5, 10, ...) above (+1) or below (-1) v.
float step_125(float v, int direction) {
    // A float just below a power of ten (0.00999999978f for 0.01) must count as that
    // power of ten, otherwise its exponent is one too low and stepping gets stuck.
    const double e = std::floor(std::log10(static_cast<double>(v) * (1.0 + 1e-5)));
    const double base = std::pow(10.0, e);
    const double m = static_cast<double>(v) / base;  // in about [1, 10)
    if (direction > 0) {
        for (double c : {1.0, 2.0, 5.0, 10.0})
            if (c > m * (1.0 + 1e-4)) return static_cast<float>(c * base);
        return static_cast<float>(10.0 * base);
    }
    double previous = 0.5;
    for (double c : {1.0, 2.0, 5.0})
        if (c < m * (1.0 - 1e-4)) previous = c;
    return static_cast<float>(previous * base);
}

// Largest 1-2-5 value that does not exceed v (tolerating float noise just below a step).
float floor_125(float v) {
    const double padded = static_cast<double>(v) * (1.0 + 1e-3);
    const double e = std::floor(std::log10(padded));
    const double base = std::pow(10.0, e);
    const double m = padded / base;
    double pick = 1.0;
    for (double c : {1.0, 2.0, 5.0})
        if (c <= m) pick = c;
    return static_cast<float>(pick * base);
}

template <class T>
T clamp_to(T v, T lo, T hi) {
    return std::min(hi, std::max(lo, v));
}

std::string number(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.4g", v);
    return buf;
}

std::string with_commas(long long n) {
    std::string s = std::to_string(n), out;
    const int len = static_cast<int>(s.size());
    for (int i = 0; i < len; ++i) {
        out += s[static_cast<std::size_t>(i)];
        if ((len - i - 1) % 3 == 0 && i != len - 1) out += ',';
    }
    return out;
}

std::string length_text(double meters) {
    char buf[32];
    if (meters < 1.0) std::snprintf(buf, sizeof buf, "%.3g mm", meters * 1000.0);
    else std::snprintf(buf, sizeof buf, "%.3g m", meters);
    return buf;
}

}  // namespace

FluidProperties fluid_properties(Fluid fluid) {
    switch (fluid) {
        case Fluid::Water: return kPresets[0];
        case Fluid::Air: return kPresets[1];
        case Fluid::Oil: return kPresets[2];
        case Fluid::Custom: break;
    }
    return {"Custom", 0.0f, 0.0f};
}

Setup Setup::defaults(CaseType type, bool gpu_available) {
    const Config c = type == CaseType::Tube ? Config::tube() : Config::cavity();
    Setup s;
    s.type = type;
    s.gpu_available = gpu_available;
    s.backend = gpu_available ? Backend::Gpu : Backend::Cpu;
    s.velocity = c.u0;
    s.width = c.length_x;
    s.height = c.length_y;
    s.nx = c.nx;
    s.ny = c.ny;
    s.sweeps = c.pressure_iterations;
    // Upwind for both systems: the thesis backward scheme diverges as soon as the flow is
    // faster than the thesis 0.01 m/s or reverses. It stays selectable.
    s.advection = Advection::Upwind;
    s.manual_dt = c.dt;
    return s;
}

Setup Setup::from_config(const Config& cfg, Backend backend, bool gpu_available) {
    Setup s = defaults(cfg.type, gpu_available);
    s.backend = (backend == Backend::Gpu && gpu_available) ? Backend::Gpu : Backend::Cpu;
    s.velocity = cfg.u0;
    s.width = cfg.length_x;
    s.height = cfg.length_y;
    s.nx = cfg.nx;
    s.ny = cfg.ny;
    s.sweeps = cfg.pressure_iterations;
    s.advection = cfg.advection;
    s.manual_dt = cfg.dt;

    s.fluid = Fluid::Custom;
    s.custom_rho = cfg.rho;
    s.custom_nu = cfg.nu;
    for (Fluid f : {Fluid::Water, Fluid::Air, Fluid::Oil}) {
        const FluidProperties p = fluid_properties(f);
        if (std::fabs(p.rho - cfg.rho) <= 1e-6f * p.rho && std::fabs(p.nu - cfg.nu) <= 1e-6f * p.nu)
            s.fluid = f;
    }

    const double dx = static_cast<double>(cfg.length_x) / cfg.nx;
    const double dy = static_cast<double>(cfg.length_y) / cfg.ny;
    s.square_cells = std::fabs(dx - dy) <= 1e-3 * dx;
    s.auto_dt = false;  // the caller gave a time step; let it switch to automatic if wanted
    s.normalize();
    return s;
}

std::vector<Field> Setup::visible_fields() const {
    std::vector<Field> f = {Field::System,      Field::Backend,     Field::Fluid,       Field::Velocity,
                            Field::Width,       Field::Height,      Field::SquareCells, Field::ResolutionX,
                            Field::ResolutionY, Field::Sweeps,      Field::Advection,   Field::TimeStepMode};
    if (!auto_dt) f.push_back(Field::TimeStep);
    return f;
}

bool Setup::editable(Field field) const {
    switch (field) {
        case Field::ResolutionY: return !square_cells;
        case Field::Backend: return gpu_available;
        default: return true;
    }
}

int Setup::derived_ny() const {
    const double ny_exact = static_cast<double>(nx) * static_cast<double>(height) / static_cast<double>(width);
    return clamp_to(static_cast<int>(std::lround(ny_exact)), kMinSide, kMaxSide);
}

void Setup::normalize() {
    velocity = clamp_to(velocity, 1.0e-4f, 10.0f);
    width = clamp_to(width, 0.01f, 100.0f);
    height = clamp_to(height, 0.005f, 50.0f);
    sweeps = clamp_to(sweeps, 1, 1000);
    manual_dt = clamp_to(manual_dt, 1.0e-6f, 1.0f);
    nx = clamp_to(nx, kMinSide, kMaxSide);
    ny = square_cells ? derived_ny() : clamp_to(ny, kMinSide, kMaxSide);

    // Keep the mesh within memory/time limits by trimming the x resolution.
    while (static_cast<long long>(nx) * ny > kMaxCells && nx > kMinSide) {
        const double shrink = std::sqrt(static_cast<double>(kMaxCells) / (static_cast<double>(nx) * ny));
        nx = std::max(kMinSide, std::min(nx - 1, static_cast<int>(nx * shrink)));
        ny = square_cells ? derived_ny() : std::max(kMinSide, static_cast<int>(ny * shrink));
    }
}

void Setup::adjust(Field field, int direction, bool coarse) {
    const int dir = direction >= 0 ? 1 : -1;
    switch (field) {
        case Field::System: {
            Setup fresh = defaults(type == CaseType::Tube ? CaseType::Cavity : CaseType::Tube, gpu_available);
            fresh.backend = backend;  // keep what is not specific to the geometry
            fresh.fluid = fluid;
            fresh.custom_rho = custom_rho;
            fresh.custom_nu = custom_nu;
            fresh.sweeps = sweeps;
            fresh.auto_dt = auto_dt;
            fresh.manual_dt = manual_dt;
            *this = fresh;
            break;
        }
        case Field::Backend:
            backend = (gpu_available && backend == Backend::Cpu) ? Backend::Gpu : Backend::Cpu;
            break;
        case Field::Fluid: {
            const int count = 3;  // presets; Custom joins the cycle only as a starting point
            int index = fluid == Fluid::Custom ? -1 : static_cast<int>(fluid);
            index = ((index + dir) % count + count) % count;
            fluid = static_cast<Fluid>(index);
            break;
        }
        case Field::Velocity: velocity = step_125(velocity, dir); break;
        case Field::Width: width = step_125(width, dir); break;
        case Field::Height: height = step_125(height, dir); break;
        case Field::SquareCells: square_cells = !square_cells; break;
        case Field::ResolutionX: nx += dir * (coarse ? 100 : 10); break;
        case Field::ResolutionY:
            if (!square_cells) ny += dir * (coarse ? 100 : 10);
            break;
        case Field::Sweeps: sweeps = static_cast<int>(std::lround(step_125(static_cast<float>(sweeps), dir))); break;
        case Field::Advection:
            advection = advection == Advection::Backward ? Advection::Upwind : Advection::Backward;
            break;
        case Field::TimeStepMode:
            if (auto_dt) manual_dt = dt();  // continue from the value that was in use
            auto_dt = !auto_dt;
            break;
        case Field::TimeStep:
            if (!auto_dt) manual_dt = step_125(manual_dt, dir);
            break;
    }
    normalize();
}

float Setup::rho() const { return fluid == Fluid::Custom ? custom_rho : fluid_properties(fluid).rho; }
float Setup::nu() const { return fluid == Fluid::Custom ? custom_nu : fluid_properties(fluid).nu; }

float Setup::dt() const {
    if (!auto_dt) return manual_dt;
    const double dx = static_cast<double>(width) / nx;
    const double dy = static_cast<double>(height) / ny;
    const double h = std::min(dx, dy);
    const double speed = std::max(std::fabs(static_cast<double>(velocity)), 1e-6);
    const double advective = 0.1 * h / speed;
    const double diffusion_rate = static_cast<double>(nu()) * (1.0 / (dx * dx) + 1.0 / (dy * dy));
    const double diffusive = diffusion_rate > 0.0 ? 0.1 / diffusion_rate : 1.0;
    const double limit = std::min({0.01, advective, diffusive});  // 0.01 s is the thesis step
    return floor_125(static_cast<float>(limit));
}

double Setup::reynolds() const {
    const double length = type == CaseType::Tube ? height : width;
    return static_cast<double>(velocity) * length / static_cast<double>(nu());
}

Config Setup::to_config() const {
    Config c;
    c.type = type;
    c.nx = nx;
    c.ny = ny;
    c.length_x = width;
    c.length_y = height;
    c.rho = rho();
    c.nu = nu();
    c.dt = dt();
    c.u0 = velocity;
    c.v0 = 0.0f;
    c.pressure_iterations = sweeps;
    c.advection = advection;
    c.warm_start_pressure = false;
    return c;
}

std::string Setup::label(Field field) const {
    const bool tube = type == CaseType::Tube;
    switch (field) {
        case Field::System: return "Simulation type";
        case Field::Backend: return "Calculate on";
        case Field::Fluid: return "Fluid";
        case Field::Velocity: return tube ? "Inlet velocity (m/s)" : "Lid velocity (m/s)";
        case Field::Width: return tube ? "Tube length (m)" : "Cavity width (m)";
        case Field::Height: return tube ? "Tube diameter (m)" : "Cavity height (m)";
        case Field::SquareCells: return "Square cells";
        case Field::ResolutionX: return "Resolution along x (cells)";
        case Field::ResolutionY: return "Resolution along y (cells)";
        case Field::Sweeps: return "Pressure sweeps per step";
        case Field::Advection: return "Convection scheme";
        case Field::TimeStepMode: return "Time step";
        case Field::TimeStep: return "   manual time step (s)";
    }
    return {};
}

std::string Setup::value_text(Field field) const {
    switch (field) {
        case Field::System:
            return type == CaseType::Tube ? "Tube (channel flow)" : "Cavity (lid-driven)";
        case Field::Backend:
            if (backend == Backend::Gpu) return "GPU (OpenCL)";
            return gpu_available ? "CPU (serial)" : "CPU (no GPU found)";
        case Field::Fluid: {
            if (fluid == Fluid::Custom) return "Custom (" + number(custom_rho) + " kg/m3)";
            const FluidProperties p = fluid_properties(fluid);
            return std::string(p.name) + " (nu " + number(p.nu) + ")";
        }
        case Field::Velocity: return number(velocity);
        case Field::Width: return number(width);
        case Field::Height: return number(height);
        case Field::SquareCells: return square_cells ? "On" : "Off";
        case Field::ResolutionX: return std::to_string(nx);
        case Field::ResolutionY: return std::to_string(ny) + (square_cells ? " (auto)" : "");
        case Field::Sweeps: return std::to_string(sweeps);
        case Field::Advection:
            return advection == Advection::Backward ? "Backward (thesis)" : "Upwind (stable)";
        case Field::TimeStepMode: return auto_dt ? "Automatic: " + number(dt()) + " s" : "Manual";
        case Field::TimeStep: return number(manual_dt);
    }
    return {};
}

std::string Setup::summary() const {
    const double dx = static_cast<double>(width) / nx;
    const double dy = static_cast<double>(height) / ny;
    char re[32];
    std::snprintf(re, sizeof re, "%.0f", reynolds());
    return std::to_string(nx) + " x " + std::to_string(ny) + " = " +
           with_commas(static_cast<long long>(nx) * ny) + " cells  |  cell " + length_text(dx) + " x " +
           length_text(dy) + "  |  dt " + number(dt()) + " s  |  Reynolds " + re;
}

std::vector<std::string> Setup::warnings() const {
    std::vector<std::string> out = to_config().warnings();
    if (advection == Advection::Backward) {
        if (type == CaseType::Cavity)
            out.emplace_back("Backward convection (thesis scheme) diverges in the cavity: choose Upwind.");
        else if (reynolds() > 1001.0)  // faster than the thesis tube
            out.emplace_back("Backward convection (thesis scheme) blows up at this speed: choose Upwind.");
    }
    if (reynolds() > 2300.0)
        out.emplace_back("Reynolds number above 2300: real flow would be turbulent, this solver is laminar only.");
    if (backend == Backend::Cpu && static_cast<long long>(nx) * ny > 1000000)
        out.emplace_back("More than 1,000,000 cells on the CPU will be slow; the GPU is much faster here.");
    return out;
}

// ---------------------------------------------------------------------------------
// typed input
// ---------------------------------------------------------------------------------

bool Setup::is_numeric(Field field) const {
    switch (field) {
        case Field::Velocity:
        case Field::Width:
        case Field::Height:
        case Field::ResolutionX:
        case Field::Sweeps:
            return true;
        case Field::ResolutionY: return !square_cells;  // derived while cells are square
        case Field::TimeStep: return !auto_dt;          // the row is hidden otherwise
        default: return false;
    }
}

std::string Setup::edit_text(Field field) const {
    char buf[32];
    switch (field) {
        case Field::Velocity: std::snprintf(buf, sizeof buf, "%.6g", static_cast<double>(velocity)); return buf;
        case Field::Width: std::snprintf(buf, sizeof buf, "%.6g", static_cast<double>(width)); return buf;
        case Field::Height: std::snprintf(buf, sizeof buf, "%.6g", static_cast<double>(height)); return buf;
        case Field::TimeStep: std::snprintf(buf, sizeof buf, "%.6g", static_cast<double>(manual_dt)); return buf;
        case Field::ResolutionX: return std::to_string(nx);
        case Field::ResolutionY: return std::to_string(ny);
        case Field::Sweeps: return std::to_string(sweeps);
        default: return {};
    }
}

std::string Setup::range_text(Field field) const {
    const std::string cells = std::to_string(kMinSide) + " to " + std::to_string(kMaxSide) +
                              " cells (at most " + with_commas(kMaxCells) + " in total)";
    switch (field) {
        case Field::Velocity: return "0.0001 to 10 m/s";
        case Field::Width: return "0.01 to 100 m";
        case Field::Height: return "0.005 to 50 m";
        case Field::ResolutionX:
        case Field::ResolutionY: return cells;
        case Field::Sweeps: return "1 to 1000";
        case Field::TimeStep: return "0.000001 to 1 s";
        default: return {};
    }
}

SetResult Setup::set_value(Field field, const std::string& text) {
    if (!is_numeric(field)) return SetResult::Invalid;

    // Spaces are ignored and a decimal comma is accepted (e.g. "0,05").
    std::string s;
    for (char c : text) {
        if (c == ' ' || c == '\t') continue;
        s += (c == ',') ? '.' : c;
    }
    if (s.empty()) return SetResult::Invalid;
    // Plain decimal numbers only: strtod would also take hex ("0x10"), "inf" and "nan".
    for (char c : s)
        if (!std::isdigit(static_cast<unsigned char>(c)) && c != '.' && c != '+' && c != '-' && c != 'e' && c != 'E')
            return SetResult::Invalid;
    char* end = nullptr;
    const double requested = std::strtod(s.c_str(), &end);
    if (*end != '\0' || !std::isfinite(requested)) return SetResult::Invalid;

    const double as_int = std::round(requested);
    // far-out values are folded into int range; normalize() then clamps to the real limits
    const auto to_int = [](double v) { return static_cast<int>(std::max(-1.0e6, std::min(1.0e6, v))); };
    switch (field) {
        case Field::Velocity: velocity = static_cast<float>(requested); break;
        case Field::Width: width = static_cast<float>(requested); break;
        case Field::Height: height = static_cast<float>(requested); break;
        case Field::TimeStep: manual_dt = static_cast<float>(requested); break;
        case Field::ResolutionX: nx = to_int(as_int); break;
        case Field::ResolutionY: ny = to_int(as_int); break;
        case Field::Sweeps: sweeps = to_int(as_int); break;
        default: return SetResult::Invalid;
    }
    normalize();

    double actual = 0.0, wanted = requested;
    switch (field) {
        case Field::Velocity: actual = velocity; break;
        case Field::Width: actual = width; break;
        case Field::Height: actual = height; break;
        case Field::TimeStep: actual = manual_dt; break;
        case Field::ResolutionX: actual = nx; wanted = as_int; break;
        case Field::ResolutionY: actual = ny; wanted = as_int; break;
        case Field::Sweeps: actual = sweeps; wanted = as_int; break;
        default: break;
    }
    const bool same = std::fabs(actual - wanted) <= 1e-4 * std::max(std::fabs(wanted), 1e-12);
    return same ? SetResult::Ok : SetResult::Limited;
}

bool Setup::same_simulation(const Setup& o) const {
    const Config a = to_config(), b = o.to_config();
    return backend == o.backend && a.type == b.type && a.nx == b.nx && a.ny == b.ny &&
           a.length_x == b.length_x && a.length_y == b.length_y && a.rho == b.rho && a.nu == b.nu &&
           a.dt == b.dt && a.u0 == b.u0 && a.pressure_iterations == b.pressure_iterations &&
           a.advection == b.advection;
}

}  // namespace tcc
