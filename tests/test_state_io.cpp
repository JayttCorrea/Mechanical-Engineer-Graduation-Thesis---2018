#include <filesystem>
#include <fstream>

#include "tcc/state_io.hpp"
#include "test_framework.hpp"

using namespace tcc;

namespace {

std::string temp_path(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

Fields sample_fields(int nx, int ny) {
    Fields f;
    f.assign(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    for (std::size_t i = 0; i < f.size(); ++i) {
        f.u[i] = static_cast<float>(i) * 0.5f;
        f.v[i] = -static_cast<float>(i);
        f.p[i] = static_cast<float>(i) * 1e-3f;
    }
    return f;
}

std::vector<char> read_all(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_all(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

TEST(state_round_trip_is_exact) {
    const std::string path = temp_path("tcc_test_roundtrip.tccs");
    const Fields f = sample_fields(7, 5);
    save_state(path, 7, 5, f);
    const SavedState s = load_state(path);
    CHECK(s.nx == 7 && s.ny == 5);
    CHECK(s.u == f.u);
    CHECK(s.v == f.v);
    CHECK(s.p == f.p);
    std::filesystem::remove(path);
}

TEST(state_save_overwrites_and_leaves_no_temp_file) {
    const std::string path = temp_path("tcc_test_overwrite.tccs");
    save_state(path, 3, 3, sample_fields(3, 3));
    save_state(path, 4, 2, sample_fields(4, 2));
    const SavedState s = load_state(path);
    CHECK(s.nx == 4 && s.ny == 2);
    CHECK(!std::filesystem::exists(path + ".tmp"));
    std::filesystem::remove(path);
}

TEST(state_load_rejects_missing_file) {
    CHECK_THROWS(load_state(temp_path("tcc_test_does_not_exist.tccs")));
}

TEST(state_load_rejects_corrupt_files) {
    const std::string path = temp_path("tcc_test_corrupt.tccs");
    save_state(path, 6, 4, sample_fields(6, 4));
    const std::vector<char> good = read_all(path);

    std::vector<char> truncated(good.begin(), good.end() - 8);
    write_all(path, truncated);
    CHECK_THROWS(load_state(path));

    std::vector<char> bad_magic = good;
    bad_magic[0] = 'X';
    write_all(path, bad_magic);
    CHECK_THROWS(load_state(path));

    std::vector<char> bad_version = good;
    bad_version[4] = 99;
    write_all(path, bad_version);
    CHECK_THROWS(load_state(path));

    // header claims a huge mesh while the file is tiny: must fail without allocating
    std::vector<char> huge = good;
    huge[8] = huge[9] = huge[10] = 0x7f;
    huge[11] = 0x00;
    write_all(path, huge);
    CHECK_THROWS(load_state(path));

    write_all(path, std::vector<char>{'T', 'C'});
    CHECK_THROWS(load_state(path));
    std::filesystem::remove(path);
}

TEST(state_save_rejects_inconsistent_sizes) {
    Fields f = sample_fields(3, 3);
    f.p.pop_back();
    CHECK_THROWS(save_state(temp_path("tcc_test_badsave.tccs"), 3, 3, f));
    CHECK_THROWS(save_state(temp_path("tcc_test_badsave.tccs"), 0, 3, Fields{}));
}
