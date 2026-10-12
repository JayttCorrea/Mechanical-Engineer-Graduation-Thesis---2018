#pragma once
// Replaces NaN / infinity in a field by the last valid value of the same cell.

#include <cmath>
#include <cstddef>
#include <vector>

namespace tcc {

class FieldGuard {
public:
    // Every non-finite entry of `values` is overwritten with the value that cell had the
    // last time it was finite (0 if it never was); finite entries are remembered for next
    // time. Returns how many entries were replaced.
    std::size_t repair(std::vector<float>& values) {
        if (last_valid_.size() != values.size()) last_valid_.assign(values.size(), 0.0f);
        std::size_t replaced = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (std::isfinite(values[i])) {
                last_valid_[i] = values[i];
            } else {
                values[i] = last_valid_[i];
                ++replaced;
            }
        }
        return replaced;
    }

    // Forget the history (after a reset the last valid value is 0 again).
    void clear() { last_valid_.clear(); }

private:
    std::vector<float> last_valid_;
};

}  // namespace tcc
