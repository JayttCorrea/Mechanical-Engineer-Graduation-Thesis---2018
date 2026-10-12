#pragma once
// Save/load of simulation state ("p" / "o" keys of the thesis program).
//
// File layout (little endian): "TCCS" | u32 version=1 | i32 nx | i32 ny |
//                              u[nx*ny] | v[nx*ny] | p[nx*ny]   (float32)

#include <string>
#include <vector>

#include "tcc/fields.hpp"

namespace tcc {

struct SavedState {
    int nx = 0;
    int ny = 0;
    std::vector<float> u, v, p;
};

// Writes to a temporary file and renames it, so a crash never leaves a torn file.
// Throws std::runtime_error on failure.
void save_state(const std::string& path, int nx, int ny, const Fields& fields);

// Throws std::runtime_error if the file is missing, truncated or not a state file.
SavedState load_state(const std::string& path);

}  // namespace tcc
