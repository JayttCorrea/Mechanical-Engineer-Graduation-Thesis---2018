cmake_minimum_required(VERSION 3.21)
project(TccFluid VERSION 1.0.0 LANGUAGES CXX
        DESCRIPTION "2D Navier-Stokes solver on CPU and GPU (Metal/OpenCL) - recreation of a graduation thesis")

include(cmake/Platform.cmake)
tcc_detect_platform()

if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)  # normalise paths when excluding system DLLs
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TCC_BUILD_GPU    "Build the GPU solver (Metal on macOS, OpenCL elsewhere)"    ON)
option(TCC_BUILD_VIEWER "Build the viewer (system GLUT on macOS, freeglut elsewhere)"   ON)
option(TCC_BUILD_TESTS  "Build the unit tests"                                        ON)
option(TCC_OPENMP       "Parallelise the CPU solver with OpenMP (thesis baseline is serial)" OFF)
option(TCC_STATIC_RUNTIME "Link the C++ runtime and freeglut into the .exe (no DLLs to ship)" ON)

option(TCC_MACOS_IPO "Enable supported link-time optimization for macOS Release builds" ON)
if(TCC_MACOS AND TCC_MACOS_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT TCC_MACOS_IPO_SUPPORTED OUTPUT TCC_IPO_ERROR LANGUAGES CXX)
    if(NOT TCC_MACOS_IPO_SUPPORTED)
        message(STATUS "macOS IPO unavailable: ${TCC_IPO_ERROR}")
    endif()
endif()

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
tcc_optimize_macos(tcc_fluid)

if(TCC_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(tcc_fluid PUBLIC OpenMP::OpenMP_CXX)
endif()

set(TCC_HAVE_OPENCL OFF)
set(TCC_HAVE_METAL OFF)
if(TCC_BUILD_GPU AND TCC_MACOS)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fluid.metal" TCC_METAL_SOURCE)
    configure_file(cmake/metal_source.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_source.hpp" @ONLY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fluid.metal")
    target_sources(tcc_fluid PRIVATE src/metal_solver.mm)
    set_source_files_properties(src/metal_solver.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_include_directories(tcc_fluid PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(tcc_fluid PRIVATE "-framework Metal" "-framework Foundation")
    target_compile_definitions(tcc_fluid PUBLIC TCC_WITH_METAL=1)
    set(TCC_HAVE_METAL ON)
elseif(TCC_BUILD_GPU)
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
message(STATUS "OpenCL solver: ${TCC_HAVE_OPENCL}; Metal solver: ${TCC_HAVE_METAL}")

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
add_executable(fluid_bench    apps/fluid_bench.cpp)
add_executable(fluid_snapshot apps/fluid_snapshot.cpp)
target_link_libraries(fluid_bench    PRIVATE tcc_fluid)
target_link_libraries(fluid_snapshot PRIVATE tcc_fluid)
tcc_optimize_macos(fluid_bench)
tcc_optimize_macos(fluid_snapshot)

if(TCC_BUILD_VIEWER)
    if(TCC_MACOS)
        find_package(GLUT REQUIRED)
        set(GLUT_INCLUDE_DIR ${GLUT_INCLUDE_DIRS})
        set(GLUT_LIBRARY GLUT::GLUT)
    else()
        find_path(GLUT_INCLUDE_DIR GL/freeglut.h)
        if(TCC_STATIC_RUNTIME)
            find_library(GLUT_LIBRARY NAMES libfreeglut.a freeglut_static freeglut glut)
        else()
            find_library(GLUT_LIBRARY NAMES freeglut glut)
        endif()
    endif()
    find_package(OpenGL)
    if(GLUT_INCLUDE_DIR AND GLUT_LIBRARY AND OPENGL_FOUND)
        if(TCC_MACOS)
            add_executable(fluid_viewer MACOSX_BUNDLE apps/fluid_viewer.cpp)
            set_target_properties(fluid_viewer PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "TCC Fluid" MACOSX_BUNDLE_GUI_IDENTIFIER "com.jaytt.tccfluid")
        else()
            add_executable(fluid_viewer apps/fluid_viewer.cpp)
        endif()
        tcc_optimize_macos(fluid_viewer)
        target_include_directories(fluid_viewer PRIVATE "${GLUT_INCLUDE_DIR}")
        target_link_libraries(fluid_viewer PRIVATE tcc_fluid "${GLUT_LIBRARY}" OpenGL::GL)
        if(WIN32 AND GLUT_LIBRARY MATCHES "[.]a$" AND NOT GLUT_LIBRARY MATCHES "[.]dll[.]a$")
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
install(TARGETS ${TCC_APPS} RUNTIME_DEPENDENCY_SET tcc_runtime_deps RUNTIME DESTINATION . BUNDLE DESTINATION .)
get_filename_component(TCC_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
if(NOT TCC_MACOS)
install(RUNTIME_DEPENDENCY_SET tcc_runtime_deps
    PRE_EXCLUDE_REGEXES "api-ms-win-.*" "ext-ms-.*"
    POST_EXCLUDE_REGEXES ".*[/\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\].*"
    DIRECTORIES "${TCC_COMPILER_DIR}"
    RUNTIME DESTINATION .)
endif()
install(FILES README.md DESTINATION .)
