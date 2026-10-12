# TCC fluid solver – CPU and GPU (OpenCL)

Recreation of the graduation thesis *"Uso de processadores gráficos para simulação de fluidos"*
(J. F. Corrêa, UEM, 2017): a 2D incompressible Navier-Stokes solver (finite differences,
pressure Poisson equation solved by Jacobi sweeps) that runs on the CPU or on the GPU through OpenCL.

Two systems are simulated, as in the thesis:

| case     | domain          | default mesh | drive                                   |
|----------|-----------------|--------------|-----------------------------------------|
| `tube`   | 2 m × 0.1 m     | 2000 × 100   | uniform inlet 0.01 m/s, free outflow    |
| `cavity` | 0.1 m × 0.1 m   | 100 × 100    | lid moving at 0.01 m/s on the top wall  |

Water-like fluid: ρ = 999 kg/m³, ν = 1e-6 m²/s, Δt = 0.01 s.

## Quick start (double-click)

Open the `dist` folder and double-click **`fluid_viewer.exe`**. No installation is needed: each `.exe`
is self-contained (C++ runtime and freeglut are linked in) and only uses DLLs that come with Windows,
so it can also be copied alone to any folder or PC.

The window has two parts that are always visible: the **settings panel** on the left and the **simulation** on the right.
Press **START** (or Enter) to run it. The panel stays open, so you can change the settings at any time and press
**RESTART** to run again with them. While the settings differ from the running simulation the button turns orange and
says **APPLY & RESTART**; **Pause / Continue** and **Defaults** are next to it. The number of pressure sweeps is the
exception: it takes effect immediately.

| setting | what you can choose |
|---|---|
| Simulation type | tube (channel flow) or lid-driven cavity |
| Calculate on | GPU (OpenCL) or CPU |
| Fluid | water, air or oil |
| Velocity | inlet speed (tube) or lid speed (cavity) |
| Size | tube length and diameter, or cavity width and height |
| Resolution | cells along x and y (or square cells: y follows x) |
| Pressure sweeps, convection scheme, time step | accuracy/stability controls (the time step is picked automatically) |

**Typing numbers.** Click a number (velocity, size, resolution, sweeps, manual time step) and type the new value, or
select the row with Up/Down and just start typing digits. Enter confirms, Esc cancels. A decimal point or comma and
exponents (`5e-3`) are accepted; values outside the allowed range are limited and the panel says so. The arrows, the
mouse wheel and Left/Right still step through convenient values (Shift = bigger resolution steps).

The text under the list shows the mesh, the time step and the Reynolds number, and warns about settings that are
unstable, turbulent or slow. **Defaults** restores
the thesis setup.

Under the simulation, buttons choose what to show (u, v, pressure, vectors) and reset the run; `s` / Space pauses and
`h` lists all keys. The window must have the focus; the console window next to it prints messages. Close the window
(or press `Esc`) to quit.

The other two programs also work with a double-click; their console stays open at the end until you press Enter:

* **`fluid_bench.exe`** runs the thesis CPU-versus-GPU experiment (seconds on a GPU machine, under a minute on CPU only) and saves `bench_results.csv`.
* **`fluid_snapshot.exe`** simulates 1000 steps of the tube and saves `snapshot_u.png`, `snapshot_v.png`, `snapshot_p.png`.

Files are written next to the executables, so keep the folder somewhere you can write to (not `C:\Program Files`).

## Programs

| program            | purpose                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `fluid_viewer.exe` | interactive window, same keys as the thesis program                     |
| `fluid_bench.exe`  | headless CPU/GPU timing (chapter 5.3 of the thesis), CSV output         |
| `fluid_snapshot.exe` | run N steps, print min/max, write PNG images of u, v, p               |

The executables import only Windows system DLLs (`KERNEL32`, `USER32`, `GDI32`, `OPENGL32`, ...).
The OpenCL kernels are embedded, and `OpenCL.dll` is not imported: it is loaded at run time from
the graphics driver. Without an OpenCL driver the programs still start and use the CPU solver.

```
fluid_viewer                                          # settings panel, press START
fluid_viewer --case cavity --nx 200                    # panel pre-filled with these values
fluid_viewer --autostart --case cavity --backend cpu   # start straight away
fluid_snapshot --case cavity --steps 20000 --ipf 50 --scale 4 --out cavity
fluid_bench --case cavity --mesh 50,100,200 --ipf 1,10,100,1000 --frames 1000 [--readback]
```
`--help` lists every option (`--nx --ny --lx --ly --rho --nu --dt --u0 --v0 --ipf --warm-start --advection`).

### Viewer keys (same as the thesis)

`s`/Space pause/continue · `w` u · `e` v · `r` pressure · `a` velocity vectors · `q`/`d` ±5 pressure sweeps (IPF) ·
`z` reset · `c` CPU↔GPU · `p` save state · `o` load state · `h` help · `Esc` quit.
Statistics are in the bar under the simulation and in the window title (the thesis reprinted the console every frame).
After START the simulation runs; `--paused` starts the first run paused like the thesis program.

## Building

Tested with MSYS2 UCRT64 (g++ 16, CMake, Ninja, `mingw-w64-ucrt-x86_64-{opencl-headers,freeglut}`).
By default everything is linked statically (`-DTCC_STATIC_RUNTIME=OFF` for DLL builds; `cmake --install`
then copies the needed DLLs beside the executables).

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix dist       # folder to hand out (the exes in build/ work too)
```
Options: `-DTCC_BUILD_GPU=OFF`, `-DTCC_BUILD_VIEWER=OFF`, `-DTCC_OPENMP=ON` (multi-threaded CPU solver;
the thesis baseline is serial, so it is off by default).

## What is different from the 2017 code

Numerics are the thesis numerics; `tests/legacy_reference.hpp` is a literal transcription of the
thesis kernels and the tests prove the new solver reproduces it step for step. Engineering changes:

* **Device-resident data.** The thesis created ~10 OpenCL buffers per kernel per frame, copied everything
  host→device→host between the three stages and never freed them. Now buffers are allocated once, pressure
  sweeps and velocity updates ping-pong between device buffers, and the host only reads data when it draws.
* **Real Jacobi.** The thesis GPU loop made the pressure input and output the same buffer after the first
  sweep (a race). Input and output buffers are always distinct.
* **2D launch** instead of decoding a 1D id with a chain of `if`s (proven equal to `i % nx`, `i / nx` by a test).
* **Settings panel** (`tcc::Setup`, unit-tested): typed numbers are parsed and range-checked, and the largest stable time step is picked for any size, speed, fluid and resolution. It defaults to upwind convection because the thesis scheme diverges in the cavity and in the tube above the thesis speed (0.05 m/s blows up); it stays selectable.
* **One drawing call** (a texture) instead of one `GL_QUAD` per cell; `glutKeyboardFunc` is no longer re-registered per cell.
* One kernel file embedded in the binary; no relative `.cl` paths, no `system("PAUSE")` / `system("cls")`.
* RAII for every OpenCL handle, error checking on every call, readable build logs, input validation.
* State files have a header (mesh size is checked on load) and are written atomically.
* `d` can no longer underflow the sweep counter; `Pmin+Pmax==0` start-up hack removed.
* **NaN / infinity guard.** Whenever the fields come back from the solver (the OpenCL readback, or the CPU
  arrays) every value is checked. A value that is not a valid number is replaced by the last valid value of
  the same cell (0 if the cell never had one), and the corrected velocity is written back to the device so the
  run continues from valid numbers. The viewer then shows a banner above the simulation, "A NaN error was fixed:
  this simulation has turbulence and might not match a real fluid", with the number of replaced values; the
  count is available as `Solver::repaired_values()`. It is a safety net, not a cure: if the setup is genuinely
  unstable the same cells are repaired again every frame and the picture stops evolving, so restart with a
  smaller speed or time step, or with upwind convection.
* Tests: 72 unit tests + 12 GPU-vs-CPU / device tests (skipped automatically without a device).

### Options that change the thesis behaviour (all default to the thesis behaviour unless noted)

* `--advection upwind`: the thesis always uses a backward difference for the convective terms, which is only
  stable for u, v > 0. It works for the tube but the cavity recirculates and **diverges** (reproduced by a test
  and by the thesis code itself). The cavity preset therefore defaults to first-order upwind; `--advection backward`
  gives the thesis scheme. The tube keeps the thesis scheme.
* `--warm-start`: keep the pressure from the previous step (the thesis zeroes it every step, so with IPF=1
  the pressure is just −B). Experimental: with all-Neumann boundaries the pressure level may drift.

## Validation against the thesis

Tube, 2000 × 100 cells, 200 s simulated (20000 steps, IPF = 50), GPU:

| quantity           | thesis (own code) | thesis (Ansys) | this program |
|--------------------|-------------------|----------------|--------------|
| max u [m/s]        | 0.01089           | 0.01155        | 0.010894     |

(The pressure range and v range depend on how long the thesis let the run settle.)
The unit tests additionally prove step-by-step agreement with the transcribed 2017 kernels for both cases.

## Benchmark (Ryzen 5 2600 serial CPU vs GTX 1660, cavity, ms per frame, 300 frames)

`readback` = fields downloaded every frame (what the thesis measured); `resident` = data stays on the GPU.

| mesh    | IPF  | CPU    | GPU readback | GPU resident |
|---------|------|--------|--------------|--------------|
| 50²     | 1    | 0.039  | 0.216        | 0.023        |
| 50²     | 1000 | 3.97   | 5.17         | 5.02         |
| 100²    | 100  | 1.70   | 0.72         | 0.49         |
| 200²    | 1    | 0.76   | 0.32         | 0.020        |
| 200²    | 1000 | 64.2   | 4.77         | 4.67         |

Same conclusion as the thesis (the GPU wins once the mesh and the work per frame are large enough; the
crossover is around 2500 cells), and keeping data on the device removes the per-frame transfer cost the thesis
identified. Full CSVs are produced by `fluid_bench`.

## Layout

```
kernels/fluid.cl      OpenCL kernels (B source term, Jacobi sweep, momentum update)
src/, include/tcc/    library: config, CPU solver, OpenCL solver, colormap, state I/O, PNG writer
apps/                 viewer, benchmark, snapshot
tests/                unit tests + legacy_reference.hpp (the 2017 code, for comparison)
```
