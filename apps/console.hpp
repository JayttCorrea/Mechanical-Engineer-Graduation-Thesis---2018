#pragma once
// Keeps the console window readable when a program is started by double-clicking it.
//
// Explorer gives a double-clicked console program its own console, which Windows closes
// the moment the program exits, taking any results or error message with it. When the
// process is the only one attached to its console we ask for Enter before closing.
// Started from cmd/PowerShell/a script, the console is shared and nothing changes.

#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace console {

// True when the program was started on its own console (double-click / shortcut).
inline bool owns_console() {
#ifdef _WIN32
    DWORD ids[2];
    return GetConsoleProcessList(ids, 2) == 1;
#else
    return false;
#endif
}

inline void wait_for_enter() {
    if (!owns_console()) return;
    std::printf("\nPress Enter to close this window...");
    std::fflush(stdout);
    int c;
    do { c = std::getchar(); } while (c != '\n' && c != EOF);
}

inline bool& pause_at_exit_flag() {
    static bool flag = false;
    return flag;
}

// While enabled, any exit (normal return, std::exit, argument errors) waits for Enter
// first if the program owns its console.
inline void pause_at_exit(bool enabled) {
    static bool registered = false;
    pause_at_exit_flag() = enabled;
    if (!registered) {
        registered = true;
        std::atexit([] {
            if (pause_at_exit_flag()) wait_for_enter();
        });
    }
}

}  // namespace console
