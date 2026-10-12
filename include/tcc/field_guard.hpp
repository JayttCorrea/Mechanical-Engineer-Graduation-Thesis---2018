#pragma once
// Replaces invalid values in a field by the last valid value of the same cell.
// "Invalid" is NaN / infinity and, optionally, anything larger than a plausibility limit
// (a diverging run produces huge but finite numbers long before the first NaN).

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tcc {

class FieldGuard {
public:
    // Every invalid entry of `values` is overwritten with the value that cell had the last
    // time it was valid (0 if it never was); valid entries are remembered for next time.
    // Returns how many entries were replaced.
    std::size_t repair(std::vector<float>& values) {
        if (last_valid_.size() != values.size()) last_valid_.assign(values.size(), 0.0f);
        std::size_t replaced = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (std::isfinite(values[i]) && std::fabs(values[i]) <= limit_) {
                last_valid_[i] = values[i];
            } else {
                values[i] = last_valid_[i];
                ++replaced;
            }
        }
        return replaced;
    }

    // Values with |x| above `limit` also count as invalid. Infinity (the default) checks
    // for NaN / infinity only.
    void set_limit(float limit) { limit_ = limit > 0.0f ? limit : std::numeric_limits<float>::infinity(); }

    // Forget the history (after a reset the last valid value is 0 again).
    void clear() { last_valid_.clear(); }

private:
    std::vector<float> last_valid_;
    float limit_ = std::numeric_limits<float>::infinity();
};

}  // namespace tcc
