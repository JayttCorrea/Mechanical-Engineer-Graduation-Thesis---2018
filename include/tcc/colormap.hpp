#pragma once
// Blue -> cyan -> green -> yellow -> red colour scale used by the thesis viewer.

#include <cstdint>
#include <vector>

namespace tcc {

struct Rgb {
    float r = 0.0f, g = 0.0f, b = 0.0f;  // each in [0, 1]
};

// Colour of `value` on a scale spanning [vmin, vmax].
// Values outside the range saturate; a flat range gives green; NaN/Inf give magenta.
Rgb colormap_jet(float value, float vmin, float vmax);

// Convert a whole field to 8-bit RGB (3 bytes per cell, same cell order as `data`).
void colorize(const std::vector<float>& data, float vmin, float vmax,
              std::vector<std::uint8_t>& rgb);

}  // namespace tcc
