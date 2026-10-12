#pragma once
// Tiny immediate-mode drawing helpers (rectangles and bitmap text) in window pixels,
// origin at the bottom-left like OpenGL itself.

#include <GL/freeglut.h>

#include <string>

namespace gl_draw {

struct Color {
    float r, g, b;
};

inline constexpr Color kWhite{1.0f, 1.0f, 1.0f};
inline constexpr Color kGrey{0.55f, 0.58f, 0.64f};
inline constexpr Color kWarn{1.0f, 0.72f, 0.28f};
inline constexpr Color kError{1.0f, 0.42f, 0.42f};

// freeglut defines the font handles as pointer casts, so they cannot be constants.
inline void* font_title() { return GLUT_BITMAP_TIMES_ROMAN_24; }
inline void* font_row() { return GLUT_BITMAP_HELVETICA_18; }
inline void* font_small() { return GLUT_BITMAP_HELVETICA_12; }

// Whole-window pixel projection; clears to `background`.
inline void begin_pixels(int w, int h, Color background) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, w, 0.0, h, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_TEXTURE_2D);
    glClearColor(background.r, background.g, background.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

inline void fill_rect(float x0, float y0, float x1, float y1, Color c) {
    glColor3f(c.r, c.g, c.b);
    glBegin(GL_QUADS);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();
}

inline void outline_rect(float x0, float y0, float x1, float y1, Color c) {
    glColor3f(c.r, c.g, c.b);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();
}

// `y` is the text baseline.
inline void text(void* font, float x, float y, const std::string& s, Color c) {
    glColor3f(c.r, c.g, c.b);
    glRasterPos2f(x, y);
    glutBitmapString(font, reinterpret_cast<const unsigned char*>(s.c_str()));
}

inline float text_width(void* font, const std::string& s) {
    return static_cast<float>(glutBitmapLength(font, reinterpret_cast<const unsigned char*>(s.c_str())));
}

}  // namespace gl_draw
