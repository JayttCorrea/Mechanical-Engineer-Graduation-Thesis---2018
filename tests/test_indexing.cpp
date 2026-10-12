// The thesis decodes a 1D work-item id into (x, y) with a chain of fix-ups
// (section 4.2). The rewrite uses i % nx and i / nx; prove they are the same.

#include "legacy_reference.hpp"
#include "test_framework.hpp"

TEST(legacy_index_decoding_equals_div_mod) {
    const int sizes[][2] = {{8, 8}, {3, 3}, {5, 11}, {17, 4}, {50, 50}, {100, 60}, {2000, 100}};
    for (const auto& s : sizes) {
        const int nx = s[0], ny = s[1];
        int mismatches = 0;
        for (int i = 0; i < nx * ny; ++i) {
            int X, Y;
            legacy::cell_xy(i, nx, ny, X, Y);
            if (X != i % nx || Y != i / nx) ++mismatches;
        }
        CHECK(mismatches == 0);
    }
}
