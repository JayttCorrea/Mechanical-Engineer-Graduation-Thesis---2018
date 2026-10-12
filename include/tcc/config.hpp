#pragma once
// Simulation parameters shared by every backend.

#include <cstddef>
#include <string>
#include <vector>

namespace tcc {

enum class Advection {
    Backward,  // thesis: always the backward difference (only stable for u, v > 0)
    Upwind     // first-order upwind: the difference follows the sign of the velocity
};

enum class CaseType {
    Tube,    // uniform inlet on the left, free outflow on the right, no-slip walls
    Cavity   // lid-driven cavity: (u0, v0) is the lid velocity on the top wall
};

const char* to_string(CaseType type);
bool parse_case(const std::string& text, CaseType& out);
const char* to_string(Advection scheme);
bool parse_advection(const std::string& text, Advection& out);

struct Config {
    CaseType type = CaseType::Tube;

    int nx = 2000;               // cells along x
    int ny = 100;                // cells along y
    float length_x = 2.0f;       // domain size [m]
    float length_y = 0.1f;

    float rho = 999.0f;          // density [kg/m^3]
    float nu = 1.0e-6f;          // kinematic viscosity [m^2/s]
    float dt = 0.01f;            // time step [s]
    float u0 = 0.01f;            // inlet velocity (tube) / lid velocity (cavity) [m/s]
    float v0 = 0.0f;

    int pressure_iterations = 1; // Jacobi sweeps per time step ("IPF" in the thesis)
    Advection advection = Advection::Backward;

    // The thesis restarts the pressure from zero every step. Keeping the previous
    // pressure converges faster, but with all-Neumann boundaries the level can drift,
    // so it is opt-in.
    bool warm_start_pressure = false;

    float dx() const { return length_x / static_cast<float>(nx); }
    float dy() const { return length_y / static_cast<float>(ny); }
    std::size_t cells() const { return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny); }

    // Dimensionless numbers of the explicit scheme (based on the inlet/lid speed).
    double cfl_number() const;
    double diffusion_number() const;

    // Empty string when the configuration is usable, otherwise the first problem found.
    std::string validate() const;
    // Non-fatal hints (e.g. a time step that is likely unstable).
    std::vector<std::string> warnings() const;

    // The two systems studied in the thesis.
    static Config tube();    // 2 m x 0.1 m, 2000x100 cells, u_in = 0.01 m/s
    static Config cavity();  // 0.1 m x 0.1 m, 100x100 cells, lid = 0.01 m/s
};

}  // namespace tcc
