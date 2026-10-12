// Headless run: advance the simulation N steps, print statistics and write
// PNG images of u, v and p (and optionally the state file the viewer can load).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "cli.hpp"
#include "console.hpp"
#include "tcc/colormap.hpp"
#include "tcc/png_writer.hpp"
#include "tcc/solver_factory.hpp"
#include "tcc/state_io.hpp"

namespace {

void usage() {
    std::printf(
        "fluid_snapshot - run N steps and write PNG images\n\n"
        "  --backend cpu|gpu      solver to use (default gpu when available)\n"
        "  --steps N              time steps to run (default 1000)\n"
        "  --out PREFIX           writes PREFIX_u.png, PREFIX_v.png, PREFIX_p.png (default snapshot)\n"
        "  --scale N              integer upscaling of the images (default 1)\n"
        "  --save FILE            also write a state file loadable by fluid_viewer\n"
        "  --load FILE            start from a state file (mesh size must match)\n"
        "%s",
        cli::config_help());
}

void write_field_png(const std::string& path, const std::vector<float>& data, int nx, int ny,
                     int scale, const tcc::Range& range) {
    std::vector<std::uint8_t> cell_rgb;
    tcc::colorize(data, range.min, range.max, cell_rgb);

    // y = 0 is the bottom of the domain but the top row of an image: flip while scaling.
    const int w = nx * scale, h = ny * scale;
    std::vector<std::uint8_t> img(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3);
    for (int py = 0; py < h; ++py) {
        const int cy = ny - 1 - py / scale;
        for (int px = 0; px < w; ++px) {
            const int cx = px / scale;
            const std::size_t src = (static_cast<std::size_t>(cy) * static_cast<std::size_t>(nx) +
                                     static_cast<std::size_t>(cx)) * 3;
            const std::size_t dst = (static_cast<std::size_t>(py) * static_cast<std::size_t>(w) +
                                     static_cast<std::size_t>(px)) * 3;
            img[dst] = cell_rgb[src];
            img[dst + 1] = cell_rgb[src + 1];
            img[dst + 2] = cell_rgb[src + 2];
        }
    }
    tcc::write_png_rgb(path, w, h, img);
}

void print_range(const char* name, const tcc::Range& r) {
    std::printf("  %-2s min %+.9g   max %+.9g\n", name, static_cast<double>(r.min),
                static_cast<double>(r.max));
}

}  // namespace

int main(int argc, char** argv) {
    console::pause_at_exit(true);  // double-clicked: keep the statistics on screen
    cli::Args args(argc, argv);
    if (args.has("help")) { usage(); return 0; }
    std::vector<std::string> known = cli::config_options();
    for (const char* k : {"backend", "steps", "out", "scale", "save", "load", "help"}) known.emplace_back(k);
    args.allow_only(known);

    try {
        const tcc::Config cfg = cli::config_from_args(args);
        const tcc::Backend backend = cli::backend_from_args(args);
        const int steps = args.integer("steps", 1000);
        const int scale = args.integer("scale", 1);
        if (steps < 0 || scale < 1) cli::Args::fail("--steps must be >= 0 and --scale >= 1");
        const std::string prefix = args.str("out", "snapshot");

        for (const std::string& w : cfg.warnings()) std::printf("warning: %s\n", w.c_str());

        auto solver = tcc::make_solver(backend, cfg);
        std::printf("%s | %s %dx%d cells, dt=%g s, ipf=%d\n", solver->backend_name().c_str(),
                    tcc::to_string(cfg.type), cfg.nx, cfg.ny, static_cast<double>(cfg.dt),
                    cfg.pressure_iterations);

        if (args.has("load")) {
            const tcc::SavedState s = tcc::load_state(args.str("load", ""));
            if (s.nx != cfg.nx || s.ny != cfg.ny)
                cli::Args::fail("state file mesh does not match --nx/--ny");
            solver->set_velocity(s.u, s.v);
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < steps; ++i) {
            solver->step();
            // Look at the result regularly (as the viewer does every frame): invalid values are
            // then replaced by a recent valid value instead of an ancient one.
            if ((i + 1) % 100 == 0) (void)solver->fields();
        }
        solver->wait();
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        const tcc::Fields& f = solver->fields();
        std::printf("%d steps (t = %.3f s simulated) in %.3f s wall time\n", steps,
                    static_cast<double>(steps) * static_cast<double>(cfg.dt), secs);
        if (solver->repaired_values() > 0)
            std::printf("A NaN error was fixed (%llu values replaced by the last valid value): this simulation\n"
                        "has turbulence and might not match a real fluid.\n",
                        static_cast<unsigned long long>(solver->repaired_values()));

        const tcc::Range ru = tcc::finite_range(f.u), rv = tcc::finite_range(f.v),
                         rp = tcc::finite_range(f.p);
        print_range("u", ru);
        print_range("v", rv);
        print_range("p", rp);

        write_field_png(prefix + "_u.png", f.u, cfg.nx, cfg.ny, scale, ru);
        write_field_png(prefix + "_v.png", f.v, cfg.nx, cfg.ny, scale, rv);
        write_field_png(prefix + "_p.png", f.p, cfg.nx, cfg.ny, scale, rp);
        std::printf("wrote %s_u.png, %s_v.png, %s_p.png\n", prefix.c_str(), prefix.c_str(),
                    prefix.c_str());

        if (args.has("save")) {
            tcc::save_state(args.str("save", ""), cfg.nx, cfg.ny, f);
            std::printf("state saved to %s\n", args.str("save", "").c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
