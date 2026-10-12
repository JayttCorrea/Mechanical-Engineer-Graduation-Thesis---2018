#pragma once
// Minimal test framework: no external dependency, works with CTest.
//   TEST(name) { CHECK(...); CHECK_NEAR(a, b, tol); REQUIRE(...); CHECK_THROWS(expr); SKIP_TEST("why"); }
// CHECK records a failure and continues; REQUIRE stops the current test.
// Exit code: 0 all passed, 1 failure, 77 everything was skipped (CTest "skipped").

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace tcc_test {

struct Case {
    const char* name;
    void (*fn)();
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

struct RequireFailed {};
struct Skipped {
    std::string reason;
};

inline int& failures_in_current_test() {
    static int n = 0;
    return n;
}

inline void fail(const char* file, int line, const std::string& what) {
    std::printf("    %s:%d: %s\n", file, line, what.c_str());
    ++failures_in_current_test();
}

inline int run_all(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int passed = 0, failed = 0, skipped = 0;
    for (const Case& c : registry()) {
        if (!filter.empty() && std::string(c.name).find(filter) == std::string::npos) continue;
        failures_in_current_test() = 0;
        std::string note;
        bool was_skipped = false;
        try {
            c.fn();
        } catch (const RequireFailed&) {
        } catch (const Skipped& s) {
            was_skipped = true;
            note = s.reason;
        } catch (const std::exception& e) {
            fail(__FILE__, __LINE__, std::string("unexpected exception: ") + e.what());
        } catch (...) {
            fail(__FILE__, __LINE__, "unexpected unknown exception");
        }
        if (was_skipped) {
            ++skipped;
            std::printf("[ SKIP ] %s (%s)\n", c.name, note.c_str());
        } else if (failures_in_current_test() == 0) {
            ++passed;
            std::printf("[  OK  ] %s\n", c.name);
        } else {
            ++failed;
            std::printf("[ FAIL ] %s\n", c.name);
        }
    }
    std::printf("\n%d passed, %d failed, %d skipped\n", passed, failed, skipped);
    if (failed > 0) return 1;
    if (passed == 0 && skipped > 0) return 77;
    return 0;
}

}  // namespace tcc_test

#define TEST(name)                                                       \
    static void name();                                                  \
    static tcc_test::Registrar registrar_##name(#name, name);            \
    static void name()

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) tcc_test::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
    } while (0)

#define REQUIRE(cond)                                                    \
    do {                                                                 \
        if (!(cond)) {                                                   \
            tcc_test::fail(__FILE__, __LINE__, "REQUIRE(" #cond ")");    \
            throw tcc_test::RequireFailed{};                             \
        }                                                                \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                            \
    do {                                                                 \
        const double va_ = static_cast<double>(a), vb_ = static_cast<double>(b); \
        const double tol_ = static_cast<double>(tol);                    \
        if (!(std::fabs(va_ - vb_) <= tol_)) {                           \
            char buf_[256];                                              \
            std::snprintf(buf_, sizeof buf_, "CHECK_NEAR(" #a ", " #b ") got %.9g vs %.9g (tol %.3g)", \
                          va_, vb_, tol_);                               \
            tcc_test::fail(__FILE__, __LINE__, buf_);                    \
        }                                                                \
    } while (0)

#define CHECK_THROWS(expr)                                               \
    do {                                                                 \
        bool threw_ = false;                                             \
        try { (void)(expr); } catch (...) { threw_ = true; }             \
        if (!threw_) tcc_test::fail(__FILE__, __LINE__, "CHECK_THROWS(" #expr ") did not throw"); \
    } while (0)

#define SKIP_TEST(reason) throw tcc_test::Skipped{reason}
