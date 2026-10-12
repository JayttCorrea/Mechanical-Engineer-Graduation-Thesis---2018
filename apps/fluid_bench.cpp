// Headless CPU/GPU benchmark (the experiment of chapter 5.3 of the thesis).
//
// Prints one CSV row per (backend, mesh, ipf):
//   backend,case,nx,ny,cells,ipf,frames,ms_per_frame
//
// By default the GPU keeps all data on the device between frames. Use --readback to
// download the fields after every frame, which is what the original program did.

#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cli.hpp"
#include "console.hpp"
#include "tcc/cpu_solver.hpp"
#include "tcc/solver_factory.hpp"

namespace {

std::vector<int> parse_int_list(const std::string& text, const char* option) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        char* end = nullptr;
        const long v = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || *end != '\0' || v < 1) cli::Args::fail(std::string("bad value in --") + option);
        out.push_back(static_cast<int>(v));
    }
    if (out.empty()) cli::Args::fail(std::string("--") + option + " needs a list like 1,10,100");
    return out;
}

void usage() {
    std::printf(
        "fluid_bench - time per frame of the CPU and GPU solvers\n\n"
        "  --backend cpu|gpu|both  what to measure (default both when a GPU exists)\n"
        "  --mesh N[,N...]         square meshes NxN (overrides --nx/--ny)\n"
        "  --ipf N[,N...]          pressure sweeps per frame to test (default 1,10,100)\n"
        "  --frames N              timed frames per run (default 1000)\n"
        "  --warmup N              untimed frames before measuring (default 10)\n"
        "  --readback              download the fields after every frame\n"
        "  --csv FILE              also write the results to FILE\n"
        "%s"
        "\nWith no options at all it runs the thesis experiment (cavity, meshes 50,100,200,\n"
        "IPF 1,10,100, 300 frames) and saves bench_results.csv.\n",
        cli::config_help());
}

double time_run(tcc::Solver& solver, int frames, int warmup, bool readback) {
    for (int i = 0; i < warmup; ++i) solver.step();
    solver.wait();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) {
        solver.step();
        if (readback) (void)solver.fields();
    }
    solver.wait();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / frames;
}

}  // namespace

int main(int argc, char** argv) {
    console::pause_at_exit(true);  // double-clicked: keep the results on screen

    // No options (e.g. double-click): run the experiment of the thesis, chapter 5.3.
    const char* defaults[] = {argv[0], "--case", "cavity", "--mesh", "50,100,200", "--ipf", "1,10,100",
                              "--frames", "300", "--csv", "bench_results.csv"};
    if (argc == 1) {
        argc = static_cast<int>(sizeof defaults / sizeof defaults[0]);
        argv = const_cast<char**>(defaults);
        std::printf("No options given: running the thesis CPU/GPU benchmark (usually under a minute).\n"
                    "Results are also saved to bench_results.csv. Run with --help for all options.\n\n");
    }
    cli::Args args(argc, argv);
    if (args.has("help")) { usage(); return 0; }
    std::vector<std::string> known = cli::config_options();
    for (const char* k : {"backend", "mesh", "frames", "warmup", "readback", "csv", "help"}) known.emplace_back(k);
    args.allow_only(known);

    try {
        // The mesh list replaces the single --nx/--ny pair.
        std::vector<std::pair<int, int>> meshes;
        tcc::Config base = cli::config_from_args(args, /*read_ipf=*/false);
        if (args.has("mesh")) {
            for (int n : parse_int_list(args.str("mesh", ""), "mesh")) meshes.emplace_back(n, n);
        } else {
            meshes.emplace_back(base.nx, base.ny);
        }
        const std::vector<int> ipfs = args.has("ipf") ? parse_int_list(args.str("ipf", ""), "ipf")
                                                      : std::vector<int>{1, 10, 100};
        const int frames = args.integer("frames", 1000);
        const int warmup = args.integer("warmup", 10);
        if (frames < 1 || warmup < 0) cli::Args::fail("--frames must be >= 1 and --warmup >= 0");
        const bool readback = args.has("readback");

        const std::string which = args.str("backend", tcc::gpu_available() ? "both" : "cpu");
        std::vector<tcc::Backend> backends;
        if (which == "cpu" || which == "both") backends.push_back(tcc::Backend::Cpu);
        if (which == "gpu" || which == "both") backends.push_back(tcc::Backend::Gpu);
        if (backends.empty()) cli::Args::fail("--backend must be cpu, gpu or both");

        std::unique_ptr<std::FILE, int (*)(std::FILE*)> csv(nullptr, &std::fclose);
        if (args.has("csv")) {
            csv.reset(std::fopen(args.str("csv", "").c_str(), "w"));
            if (!csv) cli::Args::fail("cannot write '" + args.str("csv", "") + "'");
        }
        // Same line on screen and in the CSV file.
        auto emit = [&](const std::string& line) {
            std::fputs(line.c_str(), stdout);
            std::fflush(stdout);
            if (csv) { std::fputs(line.c_str(), csv.get()); std::fflush(csv.get()); }
        };

        emit("backend,case,nx,ny,cells,ipf,frames,ms_per_frame\n");
        for (const auto& mesh : meshes) {
            for (int ipf : ipfs) {
                for (tcc::Backend backend : backends) {
                    tcc::Config cfg = base;
                    cfg.nx = mesh.first;
                    cfg.ny = mesh.second;
                    cfg.pressure_iterations = ipf;
                    const std::string problem = cfg.validate();
                    if (!problem.empty()) cli::Args::fail(problem);

                    auto solver = tcc::make_solver(backend, cfg);
                    const double ms = time_run(*solver, frames, warmup, readback);
                    char row[160];
                    std::snprintf(row, sizeof row, "%s,%s,%d,%d,%zu,%d,%d,%.6f\n", tcc::to_string(backend),
                                  tcc::to_string(cfg.type), cfg.nx, cfg.ny, cfg.cells(), ipf, frames, ms);
                    emit(row);
                }
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
