#pragma once
// Everything the user can choose before a simulation starts (the viewer's setup menu).
//
// Setup is a plain model with no GUI code: it knows which settings exist, how each one
// steps up and down, keeps the combination valid (mesh limits, square cells, a stable
// time step) and turns the result into a Config.

#include <string>
#include <vector>

#include "tcc/config.hpp"
#include "tcc/solver_factory.hpp"

namespace tcc {

enum class Fluid { Water, Air, Oil, Custom };

struct FluidProperties {
    const char* name;
    float rho;  // density [kg/m^3]
    float nu;   // kinematic viscosity [m^2/s]
};
// Custom has no fixed properties (they come from the Setup that holds it).
FluidProperties fluid_properties(Fluid fluid);

// One row of the menu.
enum class Field {
    System,        // tube / cavity
    Backend,       // GPU / CPU
    Fluid,
    Velocity,      // inlet (tube) or lid (cavity) speed
    Width,         // tube length / cavity width
    Height,        // tube diameter / cavity height
    SquareCells,   // derive the y resolution so cells stay square
    ResolutionX,
    ResolutionY,
    Sweeps,        // Jacobi pressure sweeps per step (IPF)
    Advection,
    TimeStepMode,  // automatic (stable) or manual
    TimeStep
};

// Outcome of typing a number into a field.
enum class SetResult {
    Ok,       // accepted as typed
    Limited,  // understood, but moved into the allowed range
    Invalid   // not a number (the field is left unchanged)
};

struct Setup {
    static constexpr int kMinSide = 10;          // cells per side
    static constexpr int kMaxSide = 4000;
    static constexpr long long kMaxCells = 4000000;

    CaseType type = CaseType::Tube;
    Backend backend = Backend::Cpu;
    bool gpu_available = false;  // set by the caller; the GPU row is locked to CPU without it

    Fluid fluid = Fluid::Water;
    float custom_rho = 999.0f;   // used when fluid == Custom (values given on the command line)
    float custom_nu = 1.0e-6f;

    float velocity = 0.01f;      // [m/s]
    float width = 2.0f;          // [m]
    float height = 0.1f;         // [m]
    bool square_cells = true;
    int nx = 2000;
    int ny = 100;
    int sweeps = 1;
    Advection advection = Advection::Backward;
    bool auto_dt = true;
    float manual_dt = 0.01f;

    // The thesis setups (2000x100 tube / 100x100 cavity), automatic time step.
    static Setup defaults(CaseType type, bool gpu_available);
    // Start from an existing Config (command line); dt becomes manual.
    static Setup from_config(const Config& cfg, Backend backend, bool gpu_available);

    // Rows to show, in order (TimeStep only appears in manual mode).
    std::vector<Field> visible_fields() const;
    bool editable(Field field) const;
    // Move a setting one step. direction is +1 / -1; `coarse` takes bigger steps on meshes.
    void adjust(Field field, int direction, bool coarse = false);

    // Typed input. is_numeric: the row accepts a typed value right now.
    bool is_numeric(Field field) const;
    std::string edit_text(Field field) const;   // current value as plain text, e.g. "0.01"
    std::string range_text(Field field) const;  // e.g. "0.0001 to 10 m/s"
    // Accepts "0.5", "0,5", "5e-3"; out-of-range values are clamped (Limited).
    SetResult set_value(Field field, const std::string& text);

    // True when both setups describe the same simulation (used to tell "settings changed").
    bool same_simulation(const Setup& other) const;

    std::string label(Field field) const;       // e.g. "Inlet velocity (m/s)"
    std::string value_text(Field field) const;  // e.g. "0.01 m/s"

    float rho() const;
    float nu() const;
    // Automatic: the largest 1-2-5 step (capped at the thesis 0.01 s) that keeps the
    // advection CFL and the diffusion number at or below 0.1.
    float dt() const;
    double reynolds() const;  // based on the tube diameter / cavity width
    Config to_config() const;

    // One line about the mesh, e.g. "2000 x 100 = 200,000 cells | cell 1 x 1 mm | ...".
    std::string summary() const;
    // Things worth telling the user before they press start.
    std::vector<std::string> warnings() const;

private:
    int derived_ny() const;
    void normalize();
};

}  // namespace tcc
