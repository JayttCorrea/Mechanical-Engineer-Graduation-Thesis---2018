#include <cmath>
#include <limits>

#include "tcc/colormap.hpp"
#include "test_framework.hpp"

using namespace tcc;

namespace {
void check_rgb(Rgb c, float r, float g, float b) {
    CHECK_NEAR(c.r, r, 1e-6);
    CHECK_NEAR(c.g, g, 1e-6);
    CHECK_NEAR(c.b, b, 1e-6);
}
}  // namespace

TEST(colormap_anchor_colours) {
    // min, quarter points and max of the scale [0, 4]
    check_rgb(colormap_jet(0.0f, 0.0f, 4.0f), 0, 0, 1);  // blue
    check_rgb(colormap_jet(1.0f, 0.0f, 4.0f), 0, 1, 1);  // cyan
    check_rgb(colormap_jet(2.0f, 0.0f, 4.0f), 0, 1, 0);  // green
    check_rgb(colormap_jet(3.0f, 0.0f, 4.0f), 1, 1, 0);  // yellow
    check_rgb(colormap_jet(4.0f, 0.0f, 4.0f), 1, 0, 0);  // red
}

TEST(colormap_is_continuous) {
    Rgb prev = colormap_jet(0.0f, 0.0f, 1.0f);
    for (int i = 1; i <= 1000; ++i) {
        const Rgb c = colormap_jet(static_cast<float>(i) / 1000.0f, 0.0f, 1.0f);
        CHECK(std::fabs(c.r - prev.r) < 0.01f);
        CHECK(std::fabs(c.g - prev.g) < 0.01f);
        CHECK(std::fabs(c.b - prev.b) < 0.01f);
        prev = c;
    }
}

TEST(colormap_saturates_outside_the_range) {
    check_rgb(colormap_jet(-5.0f, 0.0f, 4.0f), 0, 0, 1);
    check_rgb(colormap_jet(99.0f, 0.0f, 4.0f), 1, 0, 0);
}

TEST(colormap_handles_degenerate_input) {
    check_rgb(colormap_jet(3.0f, 3.0f, 3.0f), 0, 1, 0);  // flat field: no division by zero
    const float nan = std::numeric_limits<float>::quiet_NaN();
    check_rgb(colormap_jet(nan, 0.0f, 1.0f), 1, 0, 1);
    check_rgb(colormap_jet(std::numeric_limits<float>::infinity(), 0.0f, 1.0f), 1, 0, 1);
    // reversed range behaves like the thesis (|max - min|); must not produce NaN
    const Rgb c = colormap_jet(0.5f, 1.0f, 0.0f);
    CHECK(std::isfinite(c.r) && std::isfinite(c.g) && std::isfinite(c.b));
}

TEST(colorize_produces_three_bytes_per_cell) {
    const std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<std::uint8_t> rgb;
    colorize(data, 0.0f, 4.0f, rgb);
    REQUIRE(rgb.size() == 15);
    CHECK(rgb[0] == 0 && rgb[1] == 0 && rgb[2] == 255);      // blue
    CHECK(rgb[12] == 255 && rgb[13] == 0 && rgb[14] == 0);   // red
}
