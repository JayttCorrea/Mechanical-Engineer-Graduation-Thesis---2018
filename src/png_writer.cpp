#include "tcc/png_writer.hpp"

#include <array>
#include <fstream>
#include <stdexcept>

namespace tcc {

namespace {

const std::array<std::uint32_t, 256>& crc_table() {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    return table;
}

void put_be32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v >> 24));
    out.push_back(static_cast<std::uint8_t>(v >> 16));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
    out.push_back(static_cast<std::uint8_t>(v));
}

void put_chunk(std::vector<std::uint8_t>& out, const char type[4],
               const std::vector<std::uint8_t>& payload) {
    put_be32(out, static_cast<std::uint32_t>(payload.size()));
    const std::size_t crc_start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), payload.begin(), payload.end());
    put_be32(out, crc32(out.data() + crc_start, out.size() - crc_start));
}

}  // namespace

std::uint32_t crc32(const std::uint8_t* data, std::size_t size, std::uint32_t crc) {
    const auto& table = crc_table();
    std::uint32_t c = crc ^ 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; ++i) c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

std::uint32_t adler32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t a = 1, b = 0;
    for (std::size_t i = 0; i < size; ++i) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void write_png_rgb(const std::string& path, int width, int height,
                   const std::vector<std::uint8_t>& rgb) {
    if (width <= 0 || height <= 0 ||
        rgb.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3)
        throw std::invalid_argument("write_png_rgb: buffer size does not match the image size");

    // Raw scanlines, each prefixed with filter type 0 (None).
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
    std::vector<std::uint8_t> raw;
    raw.reserve((row_bytes + 1) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        const auto first = rgb.begin() + static_cast<std::ptrdiff_t>(row_bytes) * y;
        raw.insert(raw.end(), first, first + static_cast<std::ptrdiff_t>(row_bytes));
    }

    // zlib stream made of "stored" (uncompressed) deflate blocks.
    std::vector<std::uint8_t> z = {0x78, 0x01};
    constexpr std::size_t kMaxBlock = 65535;
    for (std::size_t pos = 0; pos < raw.size() || pos == 0;) {
        const std::size_t len = std::min(kMaxBlock, raw.size() - pos);
        const bool last = pos + len >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<std::uint8_t>(len & 0xFF));
        z.push_back(static_cast<std::uint8_t>(len >> 8));
        z.push_back(static_cast<std::uint8_t>(~len & 0xFF));
        z.push_back(static_cast<std::uint8_t>((~len >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos),
                 raw.begin() + static_cast<std::ptrdiff_t>(pos + len));
        pos += len;
        if (last) break;
    }
    put_be32(z, adler32(raw.data(), raw.size()));

    std::vector<std::uint8_t> file = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<std::uint8_t> ihdr;
    put_be32(ihdr, static_cast<std::uint32_t>(width));
    put_be32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8-bit, truecolour, deflate, no filter, no interlace
    put_chunk(file, "IHDR", ihdr);
    put_chunk(file, "IDAT", z);
    put_chunk(file, "IEND", {});

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open '" + path + "' for writing");
    out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    if (!out) throw std::runtime_error("write to '" + path + "' failed");
}

}  // namespace tcc
