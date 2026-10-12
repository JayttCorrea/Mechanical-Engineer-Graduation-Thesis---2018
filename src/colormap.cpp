#include "tcc/colormap.hpp"

#include <algorithm>
#include <cmath>

namespace tcc {

namespace {
inline float clamp01(float x) { return std::min(1.0f, std::max(0.0f, x)); }
inline std::uint8_t to_byte(float x) {
    return static_cast<std::uint8_t>(clamp01(x) * 255.0f + 0.5f);
}
}  // namespace

Rgb colormap_jet(float value, float vmin, float vmax) {
    if (!std::isfinite(value)) return {1.0f, 0.0f, 1.0f};

    // Same four linear segments as GL_Maneger::ColorInterpolation in the thesis.
    const float q = std::fabs(vmax - vmin) / 4.0f;
    if (!(q > 0.0f) || !std::isfinite(q)) return {0.0f, 1.0f, 0.0f};

    Rgb c;
    if (value >= vmin + q * 3.0f) {
        c = {1.0f, (vmax - value) / q, 0.0f};
    } else if (value >= vmin + q * 2.0f) {
        c = {(value - (vmin + 2.0f * q)) / q, 1.0f, 0.0f};
    } else if (value >= vmin + q) {
        c = {0.0f, 1.0f, ((vmin + 2.0f * q) - value) / q};
    } else {
        c = {0.0f, (value - vmin) / q, 1.0f};
    }
    return {clamp01(c.r), clamp01(c.g), clamp01(c.b)};
}

void colorize(const std::vector<float>& data, float vmin, float vmax,
              std::vector<std::uint8_t>& rgb) {
    rgb.resize(data.size() * 3);
    for (std::size_t i = 0; i < data.size(); ++i) {
        const Rgb c = colormap_jet(data[i], vmin, vmax);
        rgb[3 * i + 0] = to_byte(c.r);
        rgb[3 * i + 1] = to_byte(c.g);
        rgb[3 * i + 2] = to_byte(c.b);
    }
}

}  // namespace tcc
