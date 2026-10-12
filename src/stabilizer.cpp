#include "tcc/stabilizer.hpp"

#include <cmath>

namespace tcc {

Stabilizer::Stabilizer(float base_dt, Advection base_advection)
    : base_dt_(base_dt), base_advection_(base_advection), advection_(base_advection) {}

float Stabilizer::dt() const {
    return base_dt_ * std::ldexp(1.0f, -level_);
}

void Stabilizer::reset() {
    advection_ = base_advection_;
    level_ = 0;
    events_ = 0;
    last_repaired_ = 0;
    last_steps_ = 0;
    quiet_steps_ = 0;
}

bool Stabilizer::update(std::uint64_t repaired_total, std::uint64_t steps_total) {
    // A solver reset sends both counters back to zero.
    if (repaired_total < last_repaired_) last_repaired_ = repaired_total;
    const std::uint64_t new_steps = steps_total >= last_steps_ ? steps_total - last_steps_ : steps_total;
    last_steps_ = steps_total;

    if (repaired_total > last_repaired_) {
        last_repaired_ = repaired_total;
        ++events_;
        quiet_steps_ = 0;
        if (advection_ == Advection::Backward) {  // unstable for any dt: remove the cause first
            advection_ = Advection::Upwind;
            return true;
        }
        if (level_ < kMaxLevel) {
            ++level_;
            return true;
        }
        return false;  // already at the smallest step: nothing left to try
    }

    quiet_steps_ += new_steps;
    if (level_ > 0 && quiet_steps_ >= kRecoverySteps) {
        --level_;
        quiet_steps_ = 0;
        return true;
    }
    return false;
}

}  // namespace tcc
