#include "tcc/state_io.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace tcc {

namespace {

static_assert(sizeof(float) == 4, "state files store IEEE-754 float32");

constexpr char kMagic[4] = {'T', 'C', 'C', 'S'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderSize = 4 + 4 + 4 + 4;

bool host_is_little_endian() {
    const std::uint16_t probe = 1;
    unsigned char first;
    std::memcpy(&first, &probe, 1);
    return first == 1;
}

void write_field(std::ofstream& out, const std::vector<float>& data) {
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));
}

}  // namespace

void save_state(const std::string& path, int nx, int ny, const Fields& f) {
    if (!host_is_little_endian()) throw std::runtime_error("state files need a little-endian host");
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    if (nx <= 0 || ny <= 0 || f.u.size() != n || f.v.size() != n || f.p.size() != n)
        throw std::runtime_error("save_state: field sizes do not match the mesh");

    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open '" + tmp + "' for writing");
        const std::uint32_t version = kVersion;
        const std::int32_t w = nx, h = ny;
        out.write(kMagic, 4);
        out.write(reinterpret_cast<const char*>(&version), 4);
        out.write(reinterpret_cast<const char*>(&w), 4);
        out.write(reinterpret_cast<const char*>(&h), 4);
        write_field(out, f.u);
        write_field(out, f.v);
        write_field(out, f.p);
        out.flush();
        if (!out) throw std::runtime_error("write to '" + tmp + "' failed");
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {  // Windows refuses to rename over an existing file on some filesystems
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path, ec);
        if (ec) throw std::runtime_error("cannot replace '" + path + "': " + ec.message());
    }
}

SavedState load_state(const std::string& path) {
    if (!host_is_little_endian()) throw std::runtime_error("state files need a little-endian host");
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open '" + path + "'");
    const std::streamoff file_size = in.tellg();
    in.seekg(0);

    if (file_size < static_cast<std::streamoff>(kHeaderSize))
        throw std::runtime_error("'" + path + "' is too short to be a state file");

    char magic[4];
    std::uint32_t version = 0;
    std::int32_t w = 0, h = 0;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&w), 4);
    in.read(reinterpret_cast<char*>(&h), 4);
    if (!in || std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("'" + path + "' is not a TCC state file");
    if (version != kVersion) throw std::runtime_error("unsupported state file version");
    if (w <= 0 || h <= 0) throw std::runtime_error("state file has an invalid mesh size");

    // Validate against the real file size before allocating anything.
    const std::uint64_t n = static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h);
    const std::uint64_t expected = kHeaderSize + 3ull * n * sizeof(float);
    if (n > std::numeric_limits<std::int32_t>::max() ||
        expected != static_cast<std::uint64_t>(file_size))
        throw std::runtime_error("state file is truncated or corrupt");

    SavedState s;
    s.nx = w;
    s.ny = h;
    for (std::vector<float>* field : {&s.u, &s.v, &s.p}) {
        field->resize(static_cast<std::size_t>(n));
        in.read(reinterpret_cast<char*>(field->data()),
                static_cast<std::streamsize>(n * sizeof(float)));
    }
    if (!in) throw std::runtime_error("failed while reading '" + path + "'");
    return s;
}

}  // namespace tcc
