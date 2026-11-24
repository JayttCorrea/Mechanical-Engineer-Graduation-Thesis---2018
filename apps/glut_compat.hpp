#pragma once
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
inline void glutBitmapString(void* font, const unsigned char* s) {
    while (*s) glutBitmapCharacter(font, *s++);
}
inline int glutBitmapLength(void* font, const unsigned char* s) {
    int n = 0; while (*s) n += glutBitmapWidth(font, *s++); return n;
}
#else
#include <GL/freeglut.h>
#endif
