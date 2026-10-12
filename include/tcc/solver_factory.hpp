#pragma once

#include <memory>
#include <string>

#include "tcc/solver.hpp"

namespace tcc {

enum class Backend { Cpu, Gpu };

const char* to_string(Backend backend);
bool parse_backend(const std::string& text, Backend& out);

// True when the library was built with OpenCL and at least one device exists.
bool gpu_available();

// Creates the requested solver. Throws std::runtime_error if the GPU backend is
// requested but unavailable, std::invalid_argument for a bad configuration.
std::unique_ptr<Solver> make_solver(Backend backend, const Config& cfg);

}  // namespace tcc
