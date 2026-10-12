#pragma once
// Dependency-free PNG output (uncompressed deflate blocks) for snapshots.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tcc {

// `rgb` holds width*height*3 bytes, first row = top of the image.
// Throws std::invalid_argument on a size mismatch, std::runtime_error on I/O failure.
void write_png_rgb(const std::string& path, int width, int height,
                   const std::vector<std::uint8_t>& rgb);

// Exposed for tests.
std::uint32_t crc32(const std::uint8_t* data, std::size_t size, std::uint32_t crc = 0);
std::uint32_t adler32(const std::uint8_t* data, std::size_t size);

}  // namespace tcc
