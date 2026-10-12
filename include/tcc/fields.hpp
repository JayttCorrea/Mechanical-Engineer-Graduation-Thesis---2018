#pragma once
// Host-side copy of the simulation state.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tcc {

struct Fields {
    std::vector<float> u;  // velocity along x   [m/s]
    std::vector<float> v;  // velocity along y   [m/s]
    std::vector<float> p;  // pressure           [Pa]
    std::vector<float> b;  // Poisson source term of the last step

    void assign(std::size_t n) {
        u.assign(n, 0.0f);
        v.assign(n, 0.0f);
        p.assign(n, 0.0f);
        b.assign(n, 0.0f);
    }
    std::size_t size() const { return u.size(); }
};

struct Range {
    float min = 0.0f;
    float max = 0.0f;
};

// Min/max ignoring non-finite values. An empty or all-NaN input gives {0, 0}.
inline Range finite_range(const std::vector<float>& data) {
    bool any = false;
    Range r;
    for (float x : data) {
        if (!std::isfinite(x)) continue;
        if (!any) { r.min = r.max = x; any = true; }
        else { r.min = std::min(r.min, x); r.max = std::max(r.max, x); }
    }
    return r;
}

inline bool all_finite(const std::vector<float>& data) {
    return std::all_of(data.begin(), data.end(), [](float x) { return std::isfinite(x); });
}

}  // namespace tcc
