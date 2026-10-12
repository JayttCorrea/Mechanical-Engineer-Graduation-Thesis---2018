#include <cstring>
#include <filesystem>
#include <fstream>

#include "tcc/png_writer.hpp"
#include "test_framework.hpp"

using namespace tcc;

namespace {

std::uint32_t be32(const std::uint8_t* p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) | p[3];
}

// Reads the file back, verifies every chunk CRC and the zlib/adler framing, and
// returns the decoded (stored-block) scanline data.
struct Decoded {
    bool ok = false;
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> raw;
};

Decoded decode_png(const std::string& path) {
    Decoded d;
    std::ifstream in(path, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    static const std::uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (bytes.size() < 8 || std::memcmp(bytes.data(), sig, 8) != 0) return d;

    std::vector<std::uint8_t> idat;
    bool saw_iend = false;
    std::size_t pos = 8;
    while (pos + 12 <= bytes.size()) {
        const std::uint32_t len = be32(&bytes[pos]);
        if (pos + 12 + len > bytes.size()) return d;
        const std::uint8_t* type = &bytes[pos + 4];
        if (crc32(type, 4 + len) != be32(&bytes[pos + 8 + len])) return d;  // CRC covers type + data
        const std::uint8_t* data = &bytes[pos + 8];
        if (std::memcmp(type, "IHDR", 4) == 0) {
            d.width = be32(data);
            d.height = be32(data + 4);
            if (data[8] != 8 || data[9] != 2) return d;
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), data, data + len);
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            saw_iend = true;
        }
        pos += 12 + len;
    }
    if (!saw_iend || idat.size() < 6) return d;

    // zlib header + stored blocks + adler32
    if (idat[0] != 0x78) return d;
    std::size_t p = 2;
    bool last = false;
    while (!last) {
        if (p + 5 > idat.size()) return d;
        last = (idat[p] & 1) != 0;
        if ((idat[p] >> 1) != 0) return d;  // must be a stored block
        const std::size_t len = idat[p + 1] | (idat[p + 2] << 8);
        const std::size_t nlen = idat[p + 3] | (idat[p + 4] << 8);
        if ((len ^ 0xFFFF) != nlen || p + 5 + len > idat.size()) return d;
        d.raw.insert(d.raw.end(), idat.begin() + static_cast<std::ptrdiff_t>(p + 5),
                     idat.begin() + static_cast<std::ptrdiff_t>(p + 5 + len));
        p += 5 + len;
    }
    if (p + 4 != idat.size()) return d;
    if (adler32(d.raw.data(), d.raw.size()) != be32(&idat[p])) return d;
    d.ok = true;
    return d;
}

}  // namespace

TEST(checksum_known_vectors) {
    const char* s = "123456789";
    CHECK(crc32(reinterpret_cast<const std::uint8_t*>(s), 9) == 0xCBF43926u);
    const char* w = "Wikipedia";
    CHECK(adler32(reinterpret_cast<const std::uint8_t*>(w), 9) == 0x11E60398u);
    CHECK(adler32(nullptr, 0) == 1u);
}

TEST(png_round_trip_small_image) {
    const std::string path = (std::filesystem::temp_directory_path() / "tcc_test_small.png").string();
    const int w = 5, h = 3;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w * h * 3));
    for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>(i * 7);
    write_png_rgb(path, w, h, rgb);

    const Decoded d = decode_png(path);
    REQUIRE(d.ok);
    CHECK(d.width == 5 && d.height == 3);
    REQUIRE(d.raw.size() == static_cast<std::size_t>(h * (1 + w * 3)));
    for (int y = 0; y < h; ++y) {
        CHECK(d.raw[static_cast<std::size_t>(y * (1 + w * 3))] == 0);  // filter byte
        for (int k = 0; k < w * 3; ++k)
            CHECK(d.raw[static_cast<std::size_t>(y * (1 + w * 3) + 1 + k)] ==
                  rgb[static_cast<std::size_t>(y * w * 3 + k)]);
    }
    std::filesystem::remove(path);
}

TEST(png_large_image_spans_several_deflate_blocks) {
    const std::string path = (std::filesystem::temp_directory_path() / "tcc_test_large.png").string();
    const int w = 700, h = 300;  // 630 KB raw > 65535 bytes per stored block
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w * h * 3));
    for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>((i * 31) ^ (i >> 8));
    write_png_rgb(path, w, h, rgb);
    const Decoded d = decode_png(path);
    REQUIRE(d.ok);
    CHECK(d.raw.size() == static_cast<std::size_t>(h * (1 + w * 3)));
    std::filesystem::remove(path);
}

TEST(png_rejects_bad_arguments) {
    const std::string path = (std::filesystem::temp_directory_path() / "tcc_test_bad.png").string();
    CHECK_THROWS(write_png_rgb(path, 4, 4, std::vector<std::uint8_t>(10)));
    CHECK_THROWS(write_png_rgb(path, 0, 4, std::vector<std::uint8_t>()));
    CHECK_THROWS(write_png_rgb("Z:/definitely/not/a/dir/x.png", 1, 1, std::vector<std::uint8_t>(3)));
}
