#pragma once
// Keeps a simulation going after the solver had to repair invalid values.
//
// Replacing NaN cells only treats the symptom: the cause is still there and the field
// explodes again on the next frame. The two causes of an explicit scheme going unstable are
//   * the thesis "backward" convection, which is unstable for any time step as soon as the
//     flow reverses or gets faster than the thesis tube  -> switch to upwind convection;
//   * a time step above the CFL / diffusion limit          -> make the time step smaller.
// The Stabilizer watches the solver's repair count and asks for exactly those changes, one
// step at a time, and gives the time step back slowly once the run has been quiet.
//
// It is plain logic (no GPU, no GUI): feed it the counters, apply what it answers.

#include <cstdint>

#include "tcc/config.hpp"

namespace tcc {

class Stabilizer {
public:
    static constexpr int kMaxLevel = 10;                       // time step can shrink by 2^10
    static constexpr std::uint64_t kRecoverySteps = 20000;     // quiet steps before one level is given back

    Stabilizer(float base_dt, Advection base_advection);

    // Call after the solver's fields were read. `repaired_total` and `steps_total` are the
    // solver's own counters (repaired_values(), steps()); they may drop when the solver is
    // reset. Returns true when dt() or advection() changed and must be applied again.
    bool update(std::uint64_t repaired_total, std::uint64_t steps_total);

    float dt() const;               // base time step divided by 2^level
    Advection advection() const { return advection_; }
    int level() const { return level_; }
    bool advection_was_changed() const { return advection_ != base_advection_; }
    std::uint64_t events() const { return events_; }  // frames in which repairs were needed

    // Back to the base settings (new run, or the user reset the simulation).
    void reset();

private:
    float base_dt_;
    Advection base_advection_;
    Advection advection_;
    int level_ = 0;
    std::uint64_t events_ = 0;
    std::uint64_t last_repaired_ = 0;
    std::uint64_t last_steps_ = 0;
    std::uint64_t quiet_steps_ = 0;
};

}  // namespace tcc
