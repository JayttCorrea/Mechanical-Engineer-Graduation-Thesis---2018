#pragma once
// Tiny command line helper shared by the applications.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "tcc/config.hpp"
#include "tcc/solver_factory.hpp"

namespace cli {

class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) != 0) fail("unexpected argument '" + a + "'");
            a = a.substr(2);
            const std::size_t eq = a.find('=');
            if (eq != std::string::npos) {
                values_[a.substr(0, eq)] = a.substr(eq + 1);
            } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                values_[a] = argv[++i];
            } else {
                values_[a] = "";
            }
        }
    }

    // Abort if the user passed an option the program does not know.
    void allow_only(const std::vector<std::string>& known) const {
        for (const auto& kv : values_) {
            bool ok = false;
            for (const auto& k : known) ok = ok || k == kv.first;
            if (!ok) fail("unknown option '--" + kv.first + "'");
        }
    }

    bool has(const std::string& key) const { return values_.count(key) != 0; }

    std::string str(const std::string& key, const std::string& def) const {
        auto it = values_.find(key);
        return it == values_.end() ? def : it->second;
    }
    int integer(const std::string& key, int def) const {
        auto it = values_.find(key);
        if (it == values_.end()) return def;
        char* end = nullptr;
        const long v = std::strtol(it->second.c_str(), &end, 10);
        if (it->second.empty() || *end != '\0') fail("option '--" + key + "' needs an integer");
        return static_cast<int>(v);
    }
    float real(const std::string& key, float def) const {
        auto it = values_.find(key);
        if (it == values_.end()) return def;
        char* end = nullptr;
        const float v = std::strtof(it->second.c_str(), &end);
        if (it->second.empty() || *end != '\0') fail("option '--" + key + "' needs a number");
        return v;
    }

    [[noreturn]] static void fail(const std::string& message) {
        std::fprintf(stderr, "error: %s\n", message.c_str());
        std::exit(2);
    }

private:
    std::map<std::string, std::string> values_;
};

// Options that describe the physical problem; used by every application.
inline const std::vector<std::string>& config_options() {
    static const std::vector<std::string> opts = {"case", "nx",  "ny",  "lx", "ly", "rho",
                                                   "nu",   "dt",  "u0",  "v0", "ipf", "warm-start", "advection"};
    return opts;
}

// `read_ipf = false` lets a program interpret --ipf itself (the benchmark takes a list).
inline tcc::Config config_from_args(const Args& a, bool read_ipf = true) {
    tcc::CaseType type = tcc::CaseType::Tube;
    const std::string case_name = a.str("case", "tube");
    if (!tcc::parse_case(case_name, type)) Args::fail("--case must be 'tube' or 'cavity'");

    tcc::Config c = (type == tcc::CaseType::Tube) ? tcc::Config::tube() : tcc::Config::cavity();
    c.nx = a.integer("nx", c.nx);
    c.ny = a.integer("ny", c.ny);
    c.length_x = a.real("lx", c.length_x);
    c.length_y = a.real("ly", c.length_y);
    c.rho = a.real("rho", c.rho);
    c.nu = a.real("nu", c.nu);
    c.dt = a.real("dt", c.dt);
    c.u0 = a.real("u0", c.u0);
    c.v0 = a.real("v0", c.v0);
    if (read_ipf) c.pressure_iterations = a.integer("ipf", c.pressure_iterations);
    c.warm_start_pressure = a.has("warm-start");
    const std::string scheme = a.str("advection", tcc::to_string(c.advection));
    if (!tcc::parse_advection(scheme, c.advection)) Args::fail("--advection must be 'backward' or 'upwind'");

    const std::string problem = c.validate();
    if (!problem.empty()) Args::fail(problem);
    return c;
}

inline const char* config_help() {
    return "  --case tube|cavity     system to simulate (default tube)\n"
           "  --nx N --ny N          mesh cells (defaults: tube 2000x100, cavity 100x100)\n"
           "  --lx M --ly M          domain size in metres\n"
           "  --rho R --nu NU        density [kg/m3], kinematic viscosity [m2/s]\n"
           "  --dt S                 time step [s] (default 0.01)\n"
           "  --u0 V --v0 V          inlet (tube) / lid (cavity) velocity [m/s]\n"
           "  --ipf N                Jacobi pressure sweeps per step (default 1)\n"
           "  --warm-start           keep the pressure between steps (thesis resets it)\n"
           "  --advection backward|upwind  backward = thesis (tube default), upwind (cavity default)\n";
}

inline tcc::Backend backend_from_args(const Args& a) {
    const std::string name = a.str("backend", tcc::gpu_available() ? "gpu" : "cpu");
    tcc::Backend b = tcc::Backend::Cpu;
    if (!tcc::parse_backend(name, b)) Args::fail("--backend must be 'cpu' or 'gpu'");
    return b;
}

}  // namespace cli
