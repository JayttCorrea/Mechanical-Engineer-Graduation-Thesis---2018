// Interactive program (OpenGL / freeglut).
//
// One window, two parts that are always visible:
//   left   the settings panel (simulation type, velocity, size, resolution, ...).
//          Numbers can be typed; START / RESTART applies the settings.
//   right  the simulation, with a bar of view buttons and statistics underneath.
//
// The whole field is drawn as one textured quad instead of one GL_QUAD per cell.

#include <GL/freeglut.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cli.hpp"
#include "console.hpp"
#include "control_panel.hpp"
#include "gl_draw.hpp"
#include "tcc/colormap.hpp"
#include "tcc/setup.hpp"
#include "tcc/solver_factory.hpp"
#include "tcc/stabilizer.hpp"
#include "tcc/state_io.hpp"

namespace {

enum class Show { U = 0, V = 1, P = 2 };

constexpr int kBarHeight = 34;                   // status bar under the simulation
constexpr int kBannerHeight = 34;                // warning banner above it
constexpr int kMinSimWidth = 240;                // smallest simulation area worth showing
const gl_draw::Color kBackground{0.07f, 0.08f, 0.10f};

struct BarButton {
    float x0, x1;  // window pixels
    int id;
};

struct App {
    ControlPanel panel;
    tcc::Setup applied;  // the settings the running simulation was started with

    tcc::Config cfg;     // the running simulation
    tcc::Backend backend = tcc::Backend::Cpu;
    std::unique_ptr<tcc::Solver> solver;

    bool start_paused = false;
    bool warm_start = false;  // --warm-start has no panel row; it is kept from the command line
    bool repair_announced = false;  // the console message for repaired values was printed
    bool auto_stabilize = true;     // --no-stabilize turns the self-healing off
    float velocity_limit = 0.0f;    // values above this are invalid (0 = only NaN / infinity)
    tcc::Stabilizer stabilizer{0.01f, tcc::Advection::Upwind};  // re-armed for every run
    bool running = false;
    Show show = Show::U;
    int vector_mode = 0;  // 0 off, 1 direction only, 2 proportional to speed
    std::string state_path = "state.tccs";

    // measurements
    double total_ms = 0.0;
    std::uint64_t timed_steps = 0;
    int last_title_ms = 0;
    tcc::Range range;

    // rendering
    GLuint texture = 0;
    std::vector<std::uint8_t> rgb;
    float view_w = 1.0f, view_h = 1.0f;  // size in pixels of the area the field is drawn in
    std::vector<BarButton> bar_buttons;  // rebuilt every frame

    // automation for smoke tests
    long max_frames = -1;
    long pause_at = -1;
    int window_width = 0, window_height = 0;  // --width / --height overrides (0 = automatic)
};

App app;

void print_help() {
    std::printf(
        "\nKeys (the window must have focus):\n"
        "  settings   Up/Down select, Left/Right change, type a number, Enter = start / restart\n"
        "  s / Space  pause / continue    z  reset the simulation\n"
        "  w  show u    e  show v    r  show pressure    a  velocity vectors\n"
        "  q  +5 pressure sweeps          d  -5 pressure sweeps\n"
        "  c  switch CPU <-> GPU          p  save state     o  load state\n"
        "  h  this help                   Esc  cancel typing / quit\n\n");
}

const char* show_name(Show s) {
    switch (s) {
        case Show::U: return "u";
        case Show::V: return "v";
        case Show::P: return "pressure";
    }
    return "?";
}

// ---------------------------------------------------------------------------------
// solver / texture management
// ---------------------------------------------------------------------------------

void allocate_texture() {
    glBindTexture(GL_TEXTURE_2D, app.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, app.cfg.nx, app.cfg.ny, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
}

void reset_statistics() {
    app.total_ms = 0.0;
    app.timed_steps = 0;
    app.repair_announced = false;
}

// Called whenever a solver starts or restarts. The solver treats absurdly large velocities
// (10x the inflow or lid speed) like NaN, and the stabiliser stands ready to react to repairs.
void arm_stabilizer(tcc::Solver& solver, const tcc::Config& cfg) {
    app.stabilizer = tcc::Stabilizer(cfg.dt, cfg.advection);
    const float scale = std::max(std::fabs(cfg.u0), std::fabs(cfg.v0));
    app.velocity_limit = app.auto_stabilize && scale > 0.0f ? 10.0f * scale : 0.0f;
    solver.set_velocity_limit(app.velocity_limit);
}

// Keep the panel and the "applied" copy in step with changes made during the simulation
// (keys q / d / c), so they are not mistaken for pending edits.
void sync_sweeps(int sweeps) {
    app.panel.setup().sweeps = std::max(1, std::min(1000, sweeps));
    app.applied.sweeps = app.panel.setup().sweeps;
}

bool switch_backend(tcc::Backend backend) {
    if (!app.solver) return false;
    try {
        std::unique_ptr<tcc::Solver> fresh = tcc::make_solver(backend, app.cfg);
        const tcc::Fields& f = app.solver->fields();
        fresh->set_velocity(f.u, f.v);
        fresh->set_pressure_iterations(app.solver->config().pressure_iterations);
        fresh->set_velocity_limit(app.velocity_limit);
        fresh->set_time_step(app.stabilizer.dt());  // keep what the stabiliser has settled on
        fresh->set_advection(app.stabilizer.advection());
        app.solver = std::move(fresh);
        app.backend = backend;
        app.panel.setup().backend = backend;
        app.applied.backend = backend;
        reset_statistics();
        std::printf("solver: %s\n", app.solver->backend_name().c_str());
        return true;
    } catch (const std::exception& e) {
        std::printf("cannot use the %s solver: %s\n", tcc::to_string(backend), e.what());
        app.panel.set_status(std::string("Cannot use the ") + tcc::to_string(backend) + " solver: " + e.what(), true);
        return false;
    }
}

// Builds a solver for `cfg` and replaces the running one. Returns an error message, or
// an empty string on success (on failure the previous simulation keeps running).
std::string start_simulation(tcc::Config cfg, tcc::Backend wanted) {
    cfg.warm_start_pressure = app.warm_start;
    const std::string problem = cfg.validate();
    if (!problem.empty()) return problem;

    std::unique_ptr<tcc::Solver> solver;
    tcc::Backend used = wanted;
    std::string error;
    try {
        solver = tcc::make_solver(wanted, cfg);
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (!solver && wanted == tcc::Backend::Gpu) {
        std::printf("GPU solver failed (%s)\nusing the CPU solver instead\n", error.c_str());
        used = tcc::Backend::Cpu;
        try {
            solver = tcc::make_solver(used, cfg);
        } catch (const std::exception& e) {
            error = e.what();
        }
    }
    if (!solver) return "Cannot start: " + error;

    const bool first = !app.solver;
    app.cfg = cfg;
    app.solver = std::move(solver);
    app.backend = used;
    arm_stabilizer(*app.solver, cfg);
    reset_statistics();
    app.running = !(first && app.start_paused);
    allocate_texture();

    for (const std::string& warning : cfg.warnings()) std::printf("warning: %s\n", warning.c_str());
    std::printf("%s %dx%d on %s, dt %g s\n", tcc::to_string(cfg.type), cfg.nx, cfg.ny,
                app.solver->backend_name().c_str(), static_cast<double>(cfg.dt));
    return {};
}

// START / RESTART: apply the panel's settings.
void apply_settings() {
    tcc::Setup s = app.panel.setup();
    const std::string error = start_simulation(s.to_config(), s.backend);
    if (!error.empty()) {
        app.panel.set_status(error, true);
        return;
    }
    // A GPU request can fall back to the CPU; show what is really running.
    s.backend = app.backend;
    app.panel.setup().backend = app.backend;
    app.applied = s;
    app.panel.set_status("", false);
}

void handle_action(ControlPanel::Action action) {
    switch (action) {
        case ControlPanel::Action::None: break;
        case ControlPanel::Action::Start: apply_settings(); break;
        case ControlPanel::Action::Quit: std::exit(0);
        case ControlPanel::Action::TogglePause:
            if (app.solver) app.running = !app.running;
            break;
        case ControlPanel::Action::SweepsChanged:  // no restart needed for this one
            if (app.solver) {
                app.solver->set_pressure_iterations(app.panel.setup().sweeps);
                app.applied.sweeps = app.panel.setup().sweeps;
            }
            break;
    }
}

ControlPanel::State panel_state() {
    ControlPanel::State s;
    s.started = static_cast<bool>(app.solver);
    s.running = app.running;
    s.dirty = s.started && !app.panel.setup().same_simulation(app.applied);
    return s;
}

// ---------------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------------

void update_title() {
    if (!app.solver) {
        glutSetWindowTitle("TCC fluid simulator");
        return;
    }
    const double ms = app.timed_steps ? app.total_ms / static_cast<double>(app.timed_steps) : 0.0;
    char title[256];
    std::snprintf(title, sizeof title,
                  "%s %dx%d | %s | step %llu | ipf %d | %.3f ms/step | %s [%.4g, %.4g]%s",
                  tcc::to_string(app.cfg.type), app.cfg.nx, app.cfg.ny,
                  app.solver->backend_name().c_str(),
                  static_cast<unsigned long long>(app.solver->steps()),
                  app.solver->config().pressure_iterations, ms, show_name(app.show),
                  static_cast<double>(app.range.min), static_cast<double>(app.range.max),
                  app.running ? "" : " | PAUSED");
    glutSetWindowTitle(title);
}

void upload_field() {
    const tcc::Fields& f = app.solver->fields();
    const std::vector<float>& data = app.show == Show::U ? f.u : app.show == Show::V ? f.v : f.p;
    app.range = tcc::finite_range(data);
    tcc::colorize(data, app.range.min, app.range.max, app.rgb);
    glBindTexture(GL_TEXTURE_2D, app.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.cfg.nx, app.cfg.ny, GL_RGB, GL_UNSIGNED_BYTE,
                    app.rgb.data());
}

void draw_vectors() {
    const tcc::Fields& f = app.solver->fields();
    const int nx = app.cfg.nx, ny = app.cfg.ny;
    // About 40 arrows across the shorter side of the mesh.
    const int stride = std::max(1, std::min(nx, ny) / 40);
    float max_speed = 0.0f;
    for (std::size_t i = 0; i < f.u.size(); ++i)
        max_speed = std::max(max_speed, std::hypot(f.u[i], f.v[i]));
    if (!(max_speed > 0.0f)) return;

    const float cell_px_x = app.view_w / static_cast<float>(nx);
    const float cell_px_y = app.view_h / static_cast<float>(ny);
    const float max_len_px = 0.9f * static_cast<float>(stride) * std::min(cell_px_x, cell_px_y);

    glDisable(GL_TEXTURE_2D);
    glColor3f(0.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    for (int y = stride / 2; y < ny; y += stride) {
        for (int x = stride / 2; x < nx; x += stride) {
            const std::size_t i = static_cast<std::size_t>(y) * static_cast<std::size_t>(nx) +
                                  static_cast<std::size_t>(x);
            const float speed = std::hypot(f.u[i], f.v[i]);
            if (!(speed > 0.0f)) continue;
            const float len = max_len_px * (app.vector_mode == 1 ? 1.0f : speed / max_speed);
            const float cx = -1.0f + 2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(nx);
            const float cy = -1.0f + 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(ny);
            const float ex = cx + (f.u[i] / speed) * len * 2.0f / app.view_w;
            const float ey = cy + (f.v[i] / speed) * len * 2.0f / app.view_h;
            glVertex2f(cx, cy);
            glVertex2f(ex, ey);
        }
    }
    glEnd();
    glEnable(GL_TEXTURE_2D);
}

void pixel_projection(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, w, 0.0, h, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_TEXTURE_2D);
}

// Largest rectangle of the given shape inside the simulation area (window pixels, y up).
struct PixelRect {
    int x, y, w, h;
};

PixelRect fit_in_area(int win_w, int win_h, double shape) {
    const int margin = 8;
    const int ax = ControlPanel::kWidth + margin, ay = kBarHeight + margin;
    const int aw = std::max(1, win_w - ControlPanel::kWidth - 2 * margin);
    const int ah = std::max(1, win_h - kBarHeight - kBannerHeight - 2 * margin);
    int w = aw, h = static_cast<int>(aw / shape);
    if (h > ah) {
        h = ah;
        w = static_cast<int>(ah * shape);
    }
    w = std::max(1, w);
    h = std::max(1, h);
    return {ax + (aw - w) / 2, ay + (ah - h) / 2, w, h};
}

// Shown shape of a region: its real proportions, limited so extreme tubes stay readable.
double display_shape(float length_x, float length_y) {
    return std::min(3.0, std::max(1.0 / 3.0, static_cast<double>(length_x) / static_cast<double>(length_y)));
}

void draw_simulation_area(int win_w, int win_h) {
    using namespace gl_draw;
    if (win_w - ControlPanel::kWidth < 20) return;

    if (!app.solver) {  // nothing running yet: outline of the region the settings describe
        const tcc::Setup& s = app.panel.setup();
        const PixelRect r = fit_in_area(win_w, win_h, display_shape(s.width, s.height));
        fill_rect(static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.x + r.w),
                  static_cast<float>(r.y + r.h), {0.10f, 0.12f, 0.16f});
        outline_rect(static_cast<float>(r.x), static_cast<float>(r.y), static_cast<float>(r.x + r.w),
                     static_cast<float>(r.y + r.h), {0.35f, 0.40f, 0.50f});
        const std::string line1 = "Press START to run the simulation";
        const float cx = static_cast<float>(r.x) + static_cast<float>(r.w) / 2.0f;
        const float cy = static_cast<float>(r.y) + static_cast<float>(r.h) / 2.0f;
        text(font_row(), cx - text_width(font_row(), line1) / 2.0f, cy + 4.0f, line1, kWhite);
        const std::string line2 = s.summary();
        text(font_small(), cx - text_width(font_small(), line2) / 2.0f, cy - 20.0f, line2, kGrey);
        return;
    }

    upload_field();
    const PixelRect r = fit_in_area(win_w, win_h, display_shape(app.cfg.length_x, app.cfg.length_y));
    app.view_w = static_cast<float>(r.w);
    app.view_h = static_cast<float>(r.h);

    // The field is drawn in normalised coordinates inside its own viewport.
    glViewport(r.x, r.y, r.w, r.h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, app.texture);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();
    if (app.vector_mode != 0) draw_vectors();
    pixel_projection(win_w, win_h);
}

// Shown for as long as the run contains values that had to be repaired.
void draw_banner(int win_w, int win_h) {
    using namespace gl_draw;
    if (!app.solver || app.solver->repaired_values() == 0) return;
    const float x0 = static_cast<float>(ControlPanel::kWidth) + 8.0f, x1 = static_cast<float>(win_w) - 8.0f;
    const float y0 = static_cast<float>(win_h - kBannerHeight) + 4.0f, y1 = static_cast<float>(win_h) - 6.0f;
    fill_rect(x0, y0, x1, y1, {0.42f, 0.27f, 0.04f});
    outline_rect(x0, y0, x1, y1, kWarn);

    const std::string full = "A NaN error was fixed: this simulation has turbulence and might not match a real fluid.";
    const std::string shorter = "NaN error fixed: turbulent flow, might not match a real fluid.";
    const std::string& message = text_width(font_small(), full) + 24.0f <= x1 - x0 ? full : shorter;
    char count[48];
    std::snprintf(count, sizeof count, "   (%llu values replaced)",
                  static_cast<unsigned long long>(app.solver->repaired_values()));
    text(font_small(), x0 + 10.0f, y0 + 7.0f, message, {1.0f, 0.93f, 0.75f});
    const float after = x0 + 10.0f + text_width(font_small(), message);
    if (after + text_width(font_small(), count) + 10.0f < x1) text(font_small(), after, y0 + 7.0f, count, kWarn);
}

void draw_bar(int win_w) {
    using namespace gl_draw;
    app.bar_buttons.clear();
    const float x_start = static_cast<float>(ControlPanel::kWidth);
    fill_rect(x_start, 0.0f, static_cast<float>(win_w), static_cast<float>(kBarHeight), {0.12f, 0.14f, 0.18f});
    if (!app.solver) return;

    static const char* vector_names[] = {"vectors: off", "vectors: direction", "vectors: speed"};
    struct Item {
        std::string label;
        int id;
        bool active;
    };
    const Item items[] = {{"u", 0, app.show == Show::U},
                          {"v", 1, app.show == Show::V},
                          {"pressure", 2, app.show == Show::P},
                          {vector_names[app.vector_mode], 3, app.vector_mode != 0},
                          {"reset", 4, false}};
    float x = x_start + 8.0f;
    for (const Item& item : items) {
        const float width = text_width(font_small(), item.label) + 18.0f;
        fill_rect(x, 6.0f, x + width, 28.0f, item.active ? Color{0.20f, 0.36f, 0.62f} : Color{0.22f, 0.26f, 0.34f});
        text(font_small(), x + 9.0f, 13.0f, item.label, kWhite);
        app.bar_buttons.push_back({x, x + width, item.id});
        x += width + 6.0f;
    }

    char stats[256];
    const double ms = app.timed_steps ? app.total_ms / static_cast<double>(app.timed_steps) : 0.0;
    int len = std::snprintf(stats, sizeof stats, "step %llu   %.3f ms/step   %s [%.4g, %.4g]",
                            static_cast<unsigned long long>(app.solver->steps()), ms, show_name(app.show),
                            static_cast<double>(app.range.min), static_cast<double>(app.range.max));
    // what the stabiliser changed on its own
    if (app.stabilizer.level() > 0 && len > 0 && len < static_cast<int>(sizeof stats))
        len += std::snprintf(stats + len, sizeof stats - static_cast<std::size_t>(len), "   dt %.3g s (stabiliser)",
                             static_cast<double>(app.stabilizer.dt()));
    if (app.stabilizer.advection_was_changed() && len > 0 && len < static_cast<int>(sizeof stats))
        std::snprintf(stats + len, sizeof stats - static_cast<std::size_t>(len), "   upwind (stabiliser)");
    if (x + text_width(font_small(), stats) + 12.0f < static_cast<float>(win_w))
        text(font_small(), x + 10.0f, 13.0f, stats, kGrey);
}

// Reacts to repaired values: removes the cause of the instability so the run can go on.
void apply_stabilizer() {
    if (!app.solver || !app.auto_stabilize) return;
    const tcc::Advection before = app.solver->config().advection;
    if (!app.stabilizer.update(app.solver->repaired_values(), app.solver->steps())) return;
    app.solver->set_time_step(app.stabilizer.dt());
    app.solver->set_advection(app.stabilizer.advection());
    if (app.stabilizer.advection() != before) {
        // show what is really running
        app.panel.setup().advection = app.stabilizer.advection();
        app.applied.advection = app.stabilizer.advection();
        std::printf("stabiliser: convection switched to upwind (the thesis scheme is unstable here)\n");
    } else {
        std::printf("stabiliser: time step now %g s\n", static_cast<double>(app.stabilizer.dt()));
    }
}

void display() {
    const int w = std::max(1, glutGet(GLUT_WINDOW_WIDTH)), h = std::max(1, glutGet(GLUT_WINDOW_HEIGHT));
    if (app.solver && app.solver->repaired_values() > 0 && !app.repair_announced) {
        app.repair_announced = true;
        std::printf("A NaN error was fixed: this simulation has turbulence and might not match a real fluid.\n");
    }
    gl_draw::begin_pixels(w, h, kBackground);
    draw_simulation_area(w, h);
    apply_stabilizer();  // the fields were just read (and repaired) by draw_simulation_area
    draw_banner(w, h);
    draw_bar(w);
    app.panel.draw(h, panel_state());
    glutSwapBuffers();

    const int now = glutGet(GLUT_ELAPSED_TIME);
    if (now - app.last_title_ms > 250) {
        app.last_title_ms = now;
        update_title();
    }
}

// ---------------------------------------------------------------------------------
// simulation control
// ---------------------------------------------------------------------------------

void run_steps() {
    // Run as many steps as fit in ~15 ms before redrawing: cheap steps would otherwise be
    // limited to one per displayed frame and the flow would take minutes to develop.
    const auto frame_start = std::chrono::steady_clock::now();
    do {
        const auto t0 = std::chrono::steady_clock::now();
        app.solver->step();
        app.solver->wait();
        app.total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++app.timed_steps;

        if (app.pause_at > 0 && static_cast<long>(app.solver->steps()) >= app.pause_at) {
            app.running = false;
            std::printf("paused at step %llu: %.4f ms/step\n",
                        static_cast<unsigned long long>(app.solver->steps()),
                        app.total_ms / static_cast<double>(app.timed_steps));
        }
        if (app.max_frames > 0 && static_cast<long>(app.timed_steps) >= app.max_frames) {
            std::printf("max-frames reached: %.4f ms/step\n", app.total_ms / static_cast<double>(app.timed_steps));
            std::exit(0);
        }
    } while (app.running &&
             std::chrono::steady_clock::now() - frame_start < std::chrono::milliseconds(15));
    glutPostRedisplay();
}

void cycle_vectors() { app.vector_mode = (app.vector_mode + 1) % 3; }

void reset_simulation() {
    if (!app.solver) return;
    // The stabiliser keeps what it learned (upwind, smaller time step): a fresh start from rest
    // would only run into the same instability again.
    app.solver->reset();
    reset_statistics();
}

// Keys that are not used by the settings panel.
void simulation_key(unsigned char key) {
    if (key == 'h') {
        print_help();
        return;
    }
    if (!app.solver) return;
    switch (key) {
        case 's': case ' ': app.running = !app.running; break;
        case 'q':
            app.solver->set_pressure_iterations(app.solver->config().pressure_iterations + 5);
            sync_sweeps(app.solver->config().pressure_iterations);
            break;
        case 'd':
            app.solver->set_pressure_iterations(app.solver->config().pressure_iterations - 5);
            sync_sweeps(app.solver->config().pressure_iterations);
            break;
        case 'w': app.show = Show::U; break;
        case 'e': app.show = Show::V; break;
        case 'r': app.show = Show::P; break;
        case 'a': cycle_vectors(); break;
        case 'z': reset_simulation(); break;
        case 'c':
            switch_backend(app.backend == tcc::Backend::Cpu ? tcc::Backend::Gpu : tcc::Backend::Cpu);
            break;
        case 'p':
            try {
                tcc::save_state(app.state_path, app.cfg.nx, app.cfg.ny, app.solver->fields());
                std::printf("state saved to %s\n", app.state_path.c_str());
            } catch (const std::exception& e) {
                std::printf("save failed: %s\n", e.what());
            }
            break;
        case 'o':
            try {
                const tcc::SavedState s = tcc::load_state(app.state_path);
                if (s.nx != app.cfg.nx || s.ny != app.cfg.ny) {
                    std::printf("load failed: file is %dx%d but the mesh is %dx%d\n", s.nx, s.ny,
                                app.cfg.nx, app.cfg.ny);
                } else {
                    app.solver->set_velocity(s.u, s.v);
                    std::printf("state loaded from %s\n", app.state_path.c_str());
                }
            } catch (const std::exception& e) {
                std::printf("load failed: %s\n", e.what());
            }
            break;
        default: break;
    }
}

// ---------------------------------------------------------------------------------
// freeglut callbacks
// ---------------------------------------------------------------------------------

void idle() {
    if (!app.solver || !app.running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }
    run_steps();
}

void reshape(int w, int h) {
    // The panel needs its full size; ask for a bigger window (a few tries at most, in case
    // the screen cannot show it).
    static int attempts = 0;
    const int min_w = ControlPanel::kWidth + kMinSimWidth, min_h = ControlPanel::kHeight;
    if ((w < min_w || h < min_h) && attempts < 3) {
        ++attempts;
        glutReshapeWindow(std::max(w, min_w), std::max(h, min_h));
        return;
    }
    attempts = 0;
    glViewport(0, 0, w, h);
}

void keyboard(unsigned char key, int, int) {
    ControlPanel::Action action;
    if (app.panel.key(key, action)) handle_action(action);
    else simulation_key(key);
    glutPostRedisplay();
}

void special(int key, int, int) {
    ControlPanel::Action action;
    app.panel.special(key, glutGetModifiers(), action);
    handle_action(action);
    glutPostRedisplay();
}

void bar_click(int x, int y, int win_h) {
    if (y < win_h - kBarHeight + 6 || y > win_h - 6) return;
    for (const BarButton& b : app.bar_buttons) {
        if (static_cast<float>(x) < b.x0 || static_cast<float>(x) > b.x1) continue;
        switch (b.id) {
            case 0: app.show = Show::U; break;
            case 1: app.show = Show::V; break;
            case 2: app.show = Show::P; break;
            case 3: cycle_vectors(); break;
            case 4: reset_simulation(); break;
            default: break;
        }
    }
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        // While typing, a click anywhere confirms the value; the panel decides what else to do.
        if (x < ControlPanel::kWidth || app.panel.editing()) {
            ControlPanel::Action action;
            app.panel.mouse(x, y, (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0, action);
            handle_action(action);
        }
        if (x >= ControlPanel::kWidth) bar_click(x, y, glutGet(GLUT_WINDOW_HEIGHT));
    }
    glutPostRedisplay();
}

void mouse_wheel(int, int direction, int x, int y) {
    if (x < ControlPanel::kWidth) {
        ControlPanel::Action action;
        app.panel.wheel(direction, x, y, (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0, action);
        handle_action(action);
    }
    glutPostRedisplay();
}

void passive_motion(int x, int y) {
    app.panel.motion(x, y);
    glutPostRedisplay();
}

void usage() {
    std::printf(
        "fluid_viewer - settings panel and interactive simulation window\n\n"
        "Without options the panel opens with the thesis tube; options pre-fill it:\n\n"
        "  --autostart            start the simulation straight away (--no-menu, --start also work)\n"
        "  --backend cpu|gpu      solver (default gpu when available)\n"
        "  --width W --height H   window size in pixels\n"
        "  --state FILE           file used by the p / o keys (default state.tccs)\n"
        "  --paused               start the first simulation paused (press s to run)\n"
        "  --pause-at N           pause automatically after N steps (thesis used 1000)\n"
        "  --max-frames N         exit after N steps (for automated checks)\n"
        "  --no-stabilize         do not react to NaN repairs (no automatic upwind / smaller time step)\n"
        "%s",
        cli::config_help());
}

}  // namespace

int main(int argc, char** argv) {
    // Until the window is up, any exit is an error the user must be able to read
    // (a double-clicked console would otherwise vanish instantly).
    console::pause_at_exit(true);

    // freeglut consumes its own options first.
    glutInit(&argc, argv);
    cli::Args args(argc, argv);
    if (args.has("help")) { usage(); return 0; }
    std::vector<std::string> known = cli::config_options();
    for (const char* k : {"backend", "width", "height", "state", "start", "paused", "pause-at", "max-frames",
                          "no-menu", "autostart", "no-stabilize", "help"})
        known.emplace_back(k);
    args.allow_only(known);

    try {
        const tcc::Config cfg = cli::config_from_args(args);
        const tcc::Backend backend = cli::backend_from_args(args);
        const bool gpu_ok = tcc::gpu_available();
        const bool autostart = args.has("autostart") || args.has("no-menu") || args.has("start");

        app.state_path = args.str("state", app.state_path);
        app.start_paused = args.has("paused");
        app.auto_stabilize = !args.has("no-stabilize");
        app.warm_start = cfg.warm_start_pressure;
        app.pause_at = args.integer("pause-at", -1);
        app.max_frames = args.integer("max-frames", -1);
        app.window_width = args.integer("width", 0);
        app.window_height = args.integer("height", 0);
        if ((args.has("width") && app.window_width < 100) || (args.has("height") && app.window_height < 100))
            cli::Args::fail("window must be at least 100x100");

        app.panel.setup() = tcc::Setup::from_config(cfg, backend, gpu_ok);
        if (!args.has("dt")) app.panel.setup().auto_dt = true;
        // The panel offers the stable scheme unless asked otherwise (the thesis one diverges easily).
        if (!args.has("advection")) app.panel.setup().advection = tcc::Advection::Upwind;
        app.applied = app.panel.setup();

        // Settings panel + a simulation area that fits the screen.
        const int screen_w = glutGet(GLUT_SCREEN_WIDTH), screen_h = glutGet(GLUT_SCREEN_HEIGHT);
        int width = ControlPanel::kWidth + 880;
        int height = ControlPanel::kHeight + 20;
        if (screen_w > 0) width = std::max(ControlPanel::kWidth + kMinSimWidth,
                                           std::min(width, static_cast<int>(0.96 * screen_w)));
        if (screen_h > 0) height = std::max(ControlPanel::kHeight, std::min(height, static_cast<int>(0.94 * screen_h)));
        if (app.window_width > 0) width = app.window_width;
        if (app.window_height > 0) height = app.window_height;

        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
        glutInitWindowSize(width, height);
        glutCreateWindow("TCC fluid simulator");

        glGenTextures(1, &app.texture);
        glBindTexture(GL_TEXTURE_2D, app.texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  // rows are nx*3 bytes, not multiples of 4
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glutDisplayFunc(display);
        glutIdleFunc(idle);
        glutKeyboardFunc(keyboard);
        glutSpecialFunc(special);
        glutMouseFunc(mouse);
        glutMouseWheelFunc(mouse_wheel);
        glutPassiveMotionFunc(passive_motion);
        glutReshapeFunc(reshape);

        if (autostart) {
            const std::string error = start_simulation(cfg, backend);
            if (!error.empty()) cli::Args::fail(error);
            app.panel.setup().backend = app.backend;
            app.applied = app.panel.setup();
        } else {
            std::printf("TCC fluid simulator - choose the settings in the window, then press START.\n");
        }
        print_help();
        // From here on, closing the window or pressing ESC is a normal way to quit.
        console::pause_at_exit(false);
        glutMainLoop();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        console::wait_for_enter();
        return 1;
    }
    return 0;
}
