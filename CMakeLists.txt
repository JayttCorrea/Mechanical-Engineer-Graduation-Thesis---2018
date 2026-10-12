cmake_minimum_required(VERSION 3.21)
project(TccFluid VERSION 1.0.0 LANGUAGES CXX
        DESCRIPTION "2D Navier-Stokes solver on CPU and GPU (OpenCL) - recreation of a graduation thesis")

if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)  # normalise paths when excluding system DLLs
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TCC_BUILD_GPU    "Build the OpenCL solver (needs the OpenCL headers only)"    ON)
option(TCC_BUILD_VIEWER "Build the interactive OpenGL/GLUT viewer (needs freeglut)"   ON)
option(TCC_BUILD_TESTS  "Build the unit tests"                                        ON)
option(TCC_OPENMP       "Parallelise the CPU solver with OpenMP (thesis baseline is serial)" OFF)
option(TCC_STATIC_RUNTIME "Link the C++ runtime and freeglut into the .exe (no DLLs to ship)" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)
    # Never enable -ffast-math: CPU/GPU results are compared against each other.
elseif(MSVC)
    add_compile_options(/W4 /permissive-)
endif()
# MinGW: -static puts libstdc++, libgcc and winpthread inside every executable, so an
# .exe runs by double-click from any folder (only Windows system DLLs are imported).
if(TCC_STATIC_RUNTIME AND MINGW)
    add_link_options(-static)
endif()

# ---------------------------------------------------------------------------
# Embed kernels/fluid.cl in the binary so the executable needs no side files.
# ---------------------------------------------------------------------------
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fluid.cl" TCC_KERNEL_SOURCE)
configure_file(cmake/kernel_source.cpp.in "${CMAKE_CURRENT_BINARY_DIR}/generated/kernel_source.cpp" @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fluid.cl")

# ---------------------------------------------------------------------------
# Core library (no OpenGL, no OpenCL headers leak through the public API)
# ---------------------------------------------------------------------------
add_library(tcc_fluid STATIC
    src/config.cpp
    src/solver.cpp
    src/cpu_solver.cpp
    src/colormap.cpp
    src/state_io.cpp
    src/png_writer.cpp
    src/solver_factory.cpp
    src/setup.cpp
    src/stabilizer.cpp)
target_include_directories(tcc_fluid PUBLIC include)

if(TCC_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(tcc_fluid PUBLIC OpenMP::OpenMP_CXX)
endif()

set(TCC_HAVE_OPENCL OFF)
if(TCC_BUILD_GPU)
    # Only the headers are needed: OpenCL.dll is loaded at run time (src/opencl_loader.cpp),
    # so the programs start even on a PC without an OpenCL driver and fall back to the CPU.
    find_path(OpenCL_INCLUDE_DIR CL/cl.h)
    if(OpenCL_INCLUDE_DIR)
        set(TCC_HAVE_OPENCL ON)
        target_sources(tcc_fluid PRIVATE src/gpu_solver.cpp src/opencl_loader.cpp
                       "${CMAKE_CURRENT_BINARY_DIR}/generated/kernel_source.cpp")
        target_include_directories(tcc_fluid PRIVATE "${OpenCL_INCLUDE_DIR}")
        target_link_libraries(tcc_fluid PRIVATE ${CMAKE_DL_LIBS})
        target_compile_definitions(tcc_fluid PUBLIC TCC_WITH_OPENCL=1)
        target_compile_definitions(tcc_fluid PRIVATE CL_TARGET_OPENCL_VERSION=120)
    else()
        message(WARNING "OpenCL headers (CL/cl.h) not found: building the CPU solver only")
    endif()
endif()
message(STATUS "OpenCL solver: ${TCC_HAVE_OPENCL}")

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
add_executable(fluid_bench    apps/fluid_bench.cpp)
add_executable(fluid_snapshot apps/fluid_snapshot.cpp)
target_link_libraries(fluid_bench    PRIVATE tcc_fluid)
target_link_libraries(fluid_snapshot PRIVATE tcc_fluid)

if(TCC_BUILD_VIEWER)
    find_path(GLUT_INCLUDE_DIR GL/freeglut.h)
    if(TCC_STATIC_RUNTIME)
        find_library(GLUT_LIBRARY NAMES libfreeglut.a freeglut_static freeglut glut)
    else()
        find_library(GLUT_LIBRARY NAMES freeglut glut)
    endif()
    find_package(OpenGL)
    if(GLUT_INCLUDE_DIR AND GLUT_LIBRARY AND OPENGL_FOUND)
        add_executable(fluid_viewer apps/fluid_viewer.cpp)
        target_include_directories(fluid_viewer PRIVATE "${GLUT_INCLUDE_DIR}")
        target_link_libraries(fluid_viewer PRIVATE tcc_fluid "${GLUT_LIBRARY}" OpenGL::GL)
        if(GLUT_LIBRARY MATCHES "[.]a$" AND NOT GLUT_LIBRARY MATCHES "[.]dll[.]a$")
            # static freeglut: no libfreeglut.dll needed, but its Windows dependencies are
            target_compile_definitions(fluid_viewer PRIVATE FREEGLUT_STATIC)
            target_link_libraries(fluid_viewer PRIVATE winmm gdi32)
        endif()
        if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            # keep a console attached: the viewer prints its key help there
            target_link_options(fluid_viewer PRIVATE -mconsole)
        endif()
    else()
        message(WARNING "freeglut/OpenGL not found: fluid_viewer will not be built")
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(TCC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install = a self-contained folder. With TCC_STATIC_RUNTIME (default) the executables
# import only Windows system DLLs; otherwise every non-system DLL is copied beside them.
#   cmake --install build --prefix <dist-folder>
# ---------------------------------------------------------------------------
set(TCC_APPS fluid_bench fluid_snapshot)
if(TARGET fluid_viewer)
    list(APPEND TCC_APPS fluid_viewer)
endif()
install(TARGETS ${TCC_APPS} RUNTIME_DEPENDENCY_SET tcc_runtime_deps RUNTIME DESTINATION .)
get_filename_component(TCC_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
install(RUNTIME_DEPENDENCY_SET tcc_runtime_deps
    PRE_EXCLUDE_REGEXES "api-ms-win-.*" "ext-ms-.*"
    POST_EXCLUDE_REGEXES ".*[/\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\].*"
    DIRECTORIES "${TCC_COMPILER_DIR}"
    RUNTIME DESTINATION .)
install(FILES README.md DESTINATION .)
