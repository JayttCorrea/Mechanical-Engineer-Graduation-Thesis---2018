#pragma once
// The settings panel on the left of the window. It stays visible while the simulation runs.
//
// All state lives in tcc::Setup; this class draws it and turns input into Setup changes:
//
//   Up/Down  select a row            Left/Right or + -  change it (Shift = big steps)
//   0-9 . -  start typing a number   Enter  confirm the typed value / start or restart
//   Esc      cancel typing / quit    Tab    next row
//   Mouse    click a row, its arrows or its value; click a number to type it; wheel changes a row

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "gl_draw.hpp"
#include "tcc/setup.hpp"

class ControlPanel {
public:
    enum class Action {
        None,
        Start,          // apply the settings and (re)start the simulation
        Quit,
        TogglePause,
        SweepsChanged   // the sweep count can be applied to a running simulation
    };

    // What the viewer tells the panel about the simulation.
    struct State {
        bool started = false;  // a simulation exists
        bool running = false;
        bool dirty = false;    // the settings differ from the running simulation
    };

    static constexpr int kWidth = 580;   // panel width in pixels
    static constexpr int kHeight = 720;  // height the layout needs

    tcc::Setup& setup() { return setup_; }
    bool editing() const { return editing_; }

    // Message under the settings (cleared by the next interaction).
    void set_status(const std::string& message, bool error) {
        status_ = message;
        status_error_ = error;
    }

    // `win_h`: window height in pixels. The panel is anchored to the top-left corner.
    void draw(int win_h, const State& state) {
        using namespace gl_draw;
        const auto fields = setup_.visible_fields();
        clamp_row(fields.size());

        glPushMatrix();
        glTranslatef(0.0f, static_cast<float>(win_h - kHeight), 0.0f);

        fill(kMargin, 10, kWidth - kMargin, 700, {0.12f, 0.14f, 0.18f});
        label(font_title(), kLabelX, 42, "Fluid simulator", kWhite);
        label(font_small(), kLabelX, 60, "Change the settings, then press START / RESTART.", kGrey);

        for (std::size_t i = 0; i < fields.size(); ++i) draw_row(static_cast<int>(i), fields[i]);

        fill(kMargin + 8, 474, kWidth - kMargin - 8, 475, {0.25f, 0.28f, 0.35f});
        const tcc::Field selected = fields[static_cast<std::size_t>(row_)];
        float y = 492.0f;
        const std::string hint = editing_ ? "Type a value from " + setup_.range_text(selected) +
                                                ".  Enter = confirm,  Esc = cancel."
                                          : help(selected);
        y = paragraph(y, hint, editing_ ? Color{1.0f, 0.90f, 0.45f} : Color{0.65f, 0.80f, 1.0f}, 2) + 6.0f;
        y = paragraph(y, setup_.summary(), {0.85f, 0.90f, 0.85f}, 2) + 6.0f;
        int budget = 3;
        for (const std::string& warning : setup_.warnings()) {
            if (budget <= 0) break;
            const float next = paragraph(y, "! " + warning, kWarn, budget);
            budget -= static_cast<int>((next - y) / 15.0f + 0.5f);
            y = next;
        }
        if (!status_.empty()) paragraph(616.0f, status_, status_error_ ? kError : kWarn, 2);

        const char* start_label = !state.started ? "START" : state.dirty ? "APPLY & RESTART" : "RESTART";
        const Color start_color = state.dirty ? Color{0.78f, 0.45f, 0.08f} : Color{0.12f, 0.50f, 0.25f};
        draw_button(kStartBtn, start_label, start_color, hover_button_ == 0, true);
        draw_button(kPauseBtn, state.running ? "Pause" : "Continue", {0.22f, 0.26f, 0.34f}, hover_button_ == 1,
                    state.started);
        draw_button(kDefaultsBtn, "Defaults", {0.22f, 0.26f, 0.34f}, hover_button_ == 2, true);
        draw_button(kQuitBtn, "Quit", {0.45f, 0.18f, 0.18f}, hover_button_ == 3, true);
        label(font_small(), kLabelX, 706, "Up/Down select   Left/Right change   type digits to enter a number   Enter start/restart", kGrey);

        glPopMatrix();
    }

    // Returns true when the key was used by the panel; otherwise the viewer may use it.
    bool key(unsigned char k, Action& action) {
        action = Action::None;
        if (editing_) {
            edit_key(k, action);
            return true;
        }
        switch (k) {
            case 13: case 10: action = Action::Start; return true;
            case 27: action = Action::Quit; return true;
            case 9: move_row(+1); clear_status(); return true;
            case '+': case '=': adjust_selected(+1, false, action); return true;
            case '-': case '_': adjust_selected(-1, false, action); return true;
            default: break;
        }
        const bool numeric_start = (k >= '0' && k <= '9') || k == '.' || k == ',';
        if (numeric_start && selected_is_numeric()) {
            begin_edit(std::string(1, static_cast<char>(k)), /*replace_on_type=*/false);
            return true;
        }
        return false;
    }

    void special(int k, int modifiers, Action& action) {
        action = Action::None;
        if (editing_) {
            if (k == GLUT_KEY_UP || k == GLUT_KEY_DOWN) {
                commit_edit(action);
                move_row(k == GLUT_KEY_UP ? -1 : +1);
            }
            return;  // Left/Right do nothing while typing
        }
        clear_status();
        const bool coarse = (modifiers & GLUT_ACTIVE_SHIFT) != 0;
        switch (k) {
            case GLUT_KEY_UP: move_row(-1); break;
            case GLUT_KEY_DOWN: move_row(+1); break;
            case GLUT_KEY_LEFT: adjust_selected(-1, coarse, action); break;
            case GLUT_KEY_RIGHT: adjust_selected(+1, coarse, action); break;
            case GLUT_KEY_HOME: row_ = 0; break;
            case GLUT_KEY_END: row_ = static_cast<int>(setup_.visible_fields().size()) - 1; break;
            default: break;
        }
    }

    // (x, y): window pixels, top-left origin. `shift`: Shift held.
    void mouse(int x, int y, bool shift, Action& action) {
        action = Action::None;
        const float lx = static_cast<float>(x), ly = static_cast<float>(y);

        // A click anywhere but inside the box being edited confirms the typed value.
        if (editing_) {
            if (row_at(lx, ly) == row_ && lx >= kBoxL && lx < kBoxR) return;
            commit_edit(action);
        } else {
            clear_status();
        }

        if (inside(kStartBtn, lx, ly)) { action = Action::Start; return; }
        if (inside(kPauseBtn, lx, ly)) { action = Action::TogglePause; return; }
        if (inside(kQuitBtn, lx, ly)) { action = Action::Quit; return; }
        if (inside(kDefaultsBtn, lx, ly)) { reset_defaults(); return; }

        const int index = row_at(lx, ly);
        if (index < 0) return;
        row_ = index;
        const tcc::Field field = setup_.visible_fields()[static_cast<std::size_t>(index)];
        if (lx >= kBoxL - kArrowZone && lx < kBoxL) {
            adjust_selected(-1, shift, action);
        } else if (lx >= kBoxR && lx < kBoxR + kArrowZone) {
            adjust_selected(+1, shift, action);
        } else if (lx >= kBoxL && lx < kBoxR) {
            if (setup_.is_numeric(field)) begin_edit(setup_.edit_text(field), /*replace_on_type=*/true);
            else if (setup_.editable(field)) adjust_selected(+1, shift, action);
        }
    }

    void wheel(int direction, int x, int y, bool shift, Action& action) {
        action = Action::None;
        if (editing_) return;
        const int index = row_at(static_cast<float>(x), static_cast<float>(y));
        if (index < 0) return;
        row_ = index;
        adjust_selected(direction > 0 ? +1 : -1, shift, action);
    }

    void motion(int x, int y) {
        const float lx = static_cast<float>(x), ly = static_cast<float>(y);
        hover_ = row_at(lx, ly);
        hover_button_ = inside(kStartBtn, lx, ly)      ? 0
                        : inside(kPauseBtn, lx, ly)    ? 1
                        : inside(kDefaultsBtn, lx, ly) ? 2
                        : inside(kQuitBtn, lx, ly)     ? 3
                                                       : -1;
    }

    void reset_defaults() { setup_ = tcc::Setup::defaults(setup_.type, setup_.gpu_available); }

private:
    using Color = gl_draw::Color;
    struct Rect {
        float x0, y0, x1, y1;  // y measured from the top of the panel
    };

    static constexpr float kMargin = 10.0f;
    static constexpr float kLabelX = 26.0f;
    static constexpr float kRowTop = 74.0f, kRowH = 30.0f;
    static constexpr float kBoxL = 296.0f, kBoxR = 500.0f, kArrowZone = 36.0f;
    static constexpr Rect kStartBtn{26.0f, 646.0f, 246.0f, 686.0f};
    static constexpr Rect kPauseBtn{256.0f, 646.0f, 366.0f, 686.0f};
    static constexpr Rect kDefaultsBtn{376.0f, 646.0f, 476.0f, 686.0f};
    static constexpr Rect kQuitBtn{486.0f, 646.0f, 554.0f, 686.0f};

    tcc::Setup setup_;
    int row_ = 0;
    int hover_ = -1;
    int hover_button_ = -1;
    bool editing_ = false;
    bool replace_on_type_ = false;  // first key replaces the whole text
    std::string edit_;
    std::string status_;
    bool status_error_ = false;

    // ---- model helpers ------------------------------------------------------
    void clear_status() { status_.clear(); }

    void clamp_row(std::size_t count) { row_ = std::max(0, std::min(row_, static_cast<int>(count) - 1)); }

    void move_row(int delta) {
        const int n = static_cast<int>(setup_.visible_fields().size());
        row_ = ((row_ + delta) % n + n) % n;
    }

    tcc::Field selected_field() const {
        const auto fields = setup_.visible_fields();
        return fields[static_cast<std::size_t>(std::max(0, std::min(row_, static_cast<int>(fields.size()) - 1)))];
    }

    bool selected_is_numeric() const { return setup_.is_numeric(selected_field()); }

    void adjust_selected(int direction, bool coarse, Action& action) {
        const tcc::Field field = selected_field();
        if (!setup_.editable(field)) return;
        setup_.adjust(field, direction, coarse);
        clamp_row(setup_.visible_fields().size());  // the TimeStep row can disappear
        if (field == tcc::Field::Sweeps) action = Action::SweepsChanged;
    }

    // ---- typed input --------------------------------------------------------
    void begin_edit(const std::string& initial, bool replace_on_type) {
        editing_ = true;
        replace_on_type_ = replace_on_type;
        edit_ = initial;
        clear_status();
    }

    void commit_edit(Action& action) {
        if (!editing_) return;
        editing_ = false;
        const tcc::Field field = selected_field();
        switch (setup_.set_value(field, edit_)) {
            case tcc::SetResult::Ok:
                clear_status();
                if (field == tcc::Field::Sweeps) action = Action::SweepsChanged;
                break;
            case tcc::SetResult::Limited:
                set_status("Value limited to the allowed range: " + setup_.range_text(field), false);
                if (field == tcc::Field::Sweeps) action = Action::SweepsChanged;
                break;
            case tcc::SetResult::Invalid:
                set_status("\"" + edit_ + "\" is not a number; the value was not changed.", true);
                break;
        }
        clamp_row(setup_.visible_fields().size());
    }

    void edit_key(unsigned char k, Action& action) {
        if (k == 13 || k == 10) {
            commit_edit(action);
        } else if (k == 27) {
            editing_ = false;  // cancel
            clear_status();
        } else if (k == 9) {
            commit_edit(action);
            move_row(+1);
        } else if (k == 8) {  // backspace
            if (replace_on_type_) edit_.clear();
            else if (!edit_.empty()) edit_.pop_back();
            replace_on_type_ = false;
        } else if (is_number_char(k) && edit_.size() < 16) {
            if (replace_on_type_) edit_.clear();
            replace_on_type_ = false;
            edit_ += static_cast<char>(k);
        }
    }

    static bool is_number_char(unsigned char k) {
        return (k >= '0' && k <= '9') || k == '.' || k == ',' || k == '-' || k == '+' || k == 'e' || k == 'E';
    }

    // ---- text ---------------------------------------------------------------
    static std::string help(tcc::Field f) {
        using tcc::Field;
        switch (f) {
            case Field::System: return "Tube: fluid enters on the left and leaves on the right. Cavity: a lid slides over a closed box.";
            case Field::Backend: return "GPU uses the graphics card (OpenCL), CPU runs serially. Same results, very different speed.";
            case Field::Fluid: return "Density and viscosity of the fluid.";
            case Field::Velocity: return "Speed of the inflow (tube) or of the moving lid (cavity). Click the value to type one.";
            case Field::Width: return "Size along x of the simulated region. Click the value to type one.";
            case Field::Height: return "Size along y of the simulated region. Click the value to type one.";
            case Field::SquareCells: return "On: the y resolution follows the x resolution so every cell is square.";
            case Field::ResolutionX: return "Cells along x: more cells = finer detail but slower. Click the value to type one.";
            case Field::ResolutionY: return "Cells along y (automatic while square cells is On).";
            case Field::Sweeps: return "Jacobi sweeps solving the pressure each step. Takes effect immediately, no restart needed.";
            case Field::Advection: return "Backward is the thesis scheme (tube only). Upwind is stable for any flow, needed by the cavity.";
            case Field::TimeStepMode: return "Automatic picks the largest stable step. Manual lets you choose (can become unstable).";
            case Field::TimeStep: return "Seconds simulated per step. Too large and the solution blows up. Click the value to type one.";
        }
        return {};
    }

    static std::vector<std::string> wrap(const std::string& s, float max_width, void* font) {
        std::vector<std::string> lines;
        std::istringstream words(s);
        std::string word, line;
        while (words >> word) {
            const std::string candidate = line.empty() ? word : line + " " + word;
            if (!line.empty() && gl_draw::text_width(font, candidate) > max_width) {
                lines.push_back(line);
                line = "   " + word;
            } else {
                line = candidate;
            }
        }
        if (!line.empty()) lines.push_back(line);
        return lines;
    }

    // Draws wrapped text from baseline y; returns the baseline below the last line drawn.
    float paragraph(float y, const std::string& s, Color c, int max_lines) const {
        int count = 0;
        for (const std::string& line : wrap(s, kWidth - 2.0f * kLabelX, gl_draw::font_small())) {
            if (count++ >= max_lines) break;
            label(gl_draw::font_small(), kLabelX, y, line, c);
            y += 15.0f;
        }
        return y;
    }

    // ---- geometry / hit tests -------------------------------------------------
    static bool inside(const Rect& r, float lx, float ly) {
        return lx >= r.x0 && lx <= r.x1 && ly >= r.y0 && ly <= r.y1;
    }

    int row_at(float lx, float ly) const {
        if (lx < kMargin || lx > kWidth - kMargin || ly < kRowTop) return -1;
        const int index = static_cast<int>((ly - kRowTop) / kRowH);
        return index < static_cast<int>(setup_.visible_fields().size()) ? index : -1;
    }

    // ---- drawing (panel coordinates: y from the top) ------------------------------
    static float gl_y(float y_top) { return static_cast<float>(kHeight) - y_top; }

    static void fill(float x0, float y0, float x1, float y1, Color c) {
        gl_draw::fill_rect(x0, gl_y(y1), x1, gl_y(y0), c);
    }

    static void label(void* font, float x, float y_baseline, const std::string& s, Color c) {
        gl_draw::text(font, x, gl_y(y_baseline), s, c);
    }

    static void triangle(float cx, float cy_top, int direction, Color c) {
        glColor3f(c.r, c.g, c.b);
        glBegin(GL_TRIANGLES);
        glVertex2f(cx - 5.0f * direction, gl_y(cy_top) + 7.0f);  // base
        glVertex2f(cx - 5.0f * direction, gl_y(cy_top) - 7.0f);
        glVertex2f(cx + 5.0f * direction, gl_y(cy_top));         // tip points the way of `direction`
        glEnd();
    }

    void draw_row(int index, tcc::Field field) const {
        using namespace gl_draw;
        const float top = kRowTop + static_cast<float>(index) * kRowH;
        const bool selected = index == row_, hovered = index == hover_;
        const bool editable = setup_.editable(field);
        const bool typing = editing_ && selected;

        if (selected) fill(kMargin + 6, top, kWidth - kMargin - 6, top + kRowH - 3, {0.20f, 0.28f, 0.45f});
        else if (hovered) fill(kMargin + 6, top, kWidth - kMargin - 6, top + kRowH - 3, {0.16f, 0.19f, 0.26f});

        label(font_row(), kLabelX, top + 20, setup_.label(field), editable ? Color{0.92f, 0.94f, 0.98f} : kGrey);

        fill(kBoxL, top + 3, kBoxR, top + kRowH - 6, typing ? Color{0.16f, 0.20f, 0.10f} : Color{0.07f, 0.08f, 0.11f});
        if (typing) {
            outline_rect(kBoxL, gl_y(top + kRowH - 6), kBoxR, gl_y(top + 3), {1.0f, 0.90f, 0.45f});
            const std::string shown = edit_ + "_";
            label(font_row(), (kBoxL + kBoxR) / 2.0f - text_width(font_row(), shown) / 2.0f, top + 20, shown,
                  {1.0f, 0.90f, 0.45f});
            return;
        }
        const std::string value = setup_.value_text(field);
        label(font_row(), (kBoxL + kBoxR) / 2.0f - text_width(font_row(), value) / 2.0f, top + 20, value,
              editable ? kWhite : kGrey);
        if (editable) {
            const Color arrow = selected ? kWhite : Color{0.60f, 0.68f, 0.85f};
            triangle(kBoxL - 18, top + 14, -1, arrow);
            triangle(kBoxR + 18, top + 14, +1, arrow);
        }
    }

    void draw_button(const Rect& r, const std::string& text, Color base, bool hovered, bool enabled) const {
        using namespace gl_draw;
        const float k = (hovered && enabled) ? 1.35f : 1.0f;
        const Color c = enabled ? Color{std::min(1.0f, base.r * k), std::min(1.0f, base.g * k), std::min(1.0f, base.b * k)}
                                : Color{0.16f, 0.17f, 0.20f};
        fill(r.x0, r.y0, r.x1, r.y1, c);
        label(font_row(), (r.x0 + r.x1) / 2.0f - text_width(font_row(), text) / 2.0f, (r.y0 + r.y1) / 2.0f + 6.0f, text,
              enabled ? kWhite : kGrey);
    }
};
