#include <metal_stdlib>
using namespace metal;
// 2D incompressible Navier-Stokes, finite differences, one work-item per cell.
//
// Same maths as the thesis kernels (B_cl.cl, Pressure.cl, Velocity.cl) merged in a
// single program and launched on a 2D range instead of decoding a 1D id by hand.
//
// Layout : row-major, i = y * nx + x, y = 0 is the bottom row.
// Options: -DCASE_TUBE=1  inlet on the left (u0,v0), outflow on the right, no-slip walls
//          -DCASE_TUBE=0  lid-driven cavity: (u0,v0) is the lid velocity on the top row
// The convection scheme is a run-time argument of velocity_update (`upwind`): 0 = thesis
// backward difference whatever the sign of the velocity, 1 = first-order upwind (needed when the
// flow reverses, e.g. in the cavity). It can therefore change while a simulation is running.

#ifndef CASE_TUBE
#define CASE_TUBE 1
#endif

typedef struct { float uN, vN, uS, vS, uE, vE, uW, vW; } VelNb;

// Velocity of the four neighbours of cell (x, y) with boundary conditions applied.
inline VelNb velocity_neighbours(device const float* U,
                                 device const float* V,
                                 int x, int y, int nx, int ny, float u0, float v0)
{
    const int i = y * nx + x;
    VelNb n;
    if (y == ny - 1) {
#if CASE_TUBE
        n.uN = 0.0f; n.vN = 0.0f;
#else
        n.uN = u0;   n.vN = v0;
#endif
    } else { n.uN = U[i + nx]; n.vN = V[i + nx]; }

    if (y == 0) { n.uS = 0.0f; n.vS = 0.0f; }
    else        { n.uS = U[i - nx]; n.vS = V[i - nx]; }

    if (x == nx - 1) {
#if CASE_TUBE
        n.uE = U[i - 1]; n.vE = V[i - 1];   // free outflow: copy the inner cell
#else
        n.uE = 0.0f;     n.vE = 0.0f;
#endif
    } else { n.uE = U[i + 1]; n.vE = V[i + 1]; }

    if (x == 0) {
#if CASE_TUBE
        n.uW = u0;   n.vW = v0;             // inlet
#else
        n.uW = 0.0f; n.vW = 0.0f;
#endif
    } else { n.uW = U[i - 1]; n.vW = V[i - 1]; }
    return n;
}

typedef struct { float N, S, E, W; } PresNb;

// Pressure of the four neighbours: zero normal gradient on every boundary.
inline PresNb pressure_neighbours(device const float* P,
                                  int x, int y, int nx, int ny)
{
    const int i = y * nx + x;
    PresNb n;
    n.N = (y == ny - 1) ? P[i - nx] : P[i + nx];
    n.S = (y == 0)      ? P[i + nx] : P[i - nx];
    n.E = (x == nx - 1) ? P[i - 1]  : P[i + 1];
#if CASE_TUBE
    n.W = (x == 0) ? P[i]     : P[i - 1];   // thesis tube kernel: ghost = the cell itself
#else
    n.W = (x == 0) ? P[i + 1] : P[i - 1];
#endif
    return n;
}

// Step 1: source term of the pressure Poisson equation.
kernel void compute_b(device float* B [[buffer(0)]],
device const float* U [[buffer(1)]],
device const float* V [[buffer(2)]],
constant int& nx [[buffer(3)]],
constant int& ny [[buffer(4)]],
constant float& dx [[buffer(5)]],
constant float& dy [[buffer(6)]],
constant float& rho [[buffer(7)]],
constant float& dt [[buffer(8)]],
constant float& u0 [[buffer(9)]],
constant float& v0 [[buffer(10)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    const VelNb n = velocity_neighbours(U, V, x, y, nx, ny, u0, v0);

    const float Ab = (n.uE - n.uW) / (2.0f * dx);
    const float Bb = (n.vN - n.vS) / (2.0f * dy);
    const float Cb = ((n.uE - n.uW) / (2.0f * dx)) * ((n.uE - n.uW) / (2.0f * dx));
    const float Db = 2.0f * ((n.uN - n.uS) / (2.0f * dy) * (n.vE - n.vW) / (2.0f * dx));
    const float Eb = ((n.vN - n.vS) / (2.0f * dy)) * ((n.vN - n.vS) / (2.0f * dy));
    const float Fp = (dx * dx) * (dy * dy) / (2.0f * ((dx * dx) + (dy * dy)));

    B[y * nx + x] = rho * Fp * ((1.0f / dt) * (Ab + Bb) - Cb - Db - Eb);
}

// Step 2: one Jacobi sweep of the pressure Poisson equation (Pin -> Pout, no aliasing).
kernel void pressure_iterate(device const float* B [[buffer(0)]],
device const float* Pin [[buffer(1)]],
device float* Pout [[buffer(2)]],
constant int& nx [[buffer(3)]],
constant int& ny [[buffer(4)]],
constant float& dx [[buffer(5)]],
constant float& dy [[buffer(6)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    const PresNb p = pressure_neighbours(Pin, x, y, nx, ny);

    const float Ap = (p.E + p.W) * (dy * dy);
    const float Bp = (p.N + p.S) * (dx * dx);
    const float Cp = 2.0f * ((dx * dx) + (dy * dy));
    const int i = y * nx + x;
    Pout[i] = ((Ap + Bp) / Cp) - B[i];
}

// Step 3: explicit momentum update (upwind advection, central pressure and diffusion).
kernel void velocity_update(device const float* U [[buffer(0)]],
device const float* V [[buffer(1)]],
device const float* P [[buffer(2)]],
device float* Uout [[buffer(3)]],
device float* Vout [[buffer(4)]],
constant int& nx [[buffer(5)]],
constant int& ny [[buffer(6)]],
constant float& dx [[buffer(7)]],
constant float& dy [[buffer(8)]],
constant float& rho [[buffer(9)]],
constant float& nu [[buffer(10)]],
constant float& dt [[buffer(11)]],
constant float& u0 [[buffer(12)]],
constant float& v0 [[buffer(13)]],
constant int& upwind [[buffer(14)]], uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    const int i = y * nx + x;
    const VelNb n = velocity_neighbours(U, V, x, y, nx, ny, u0, v0);
    const PresNb p = pressure_neighbours(P, x, y, nx, ny);

    const float u = U[i], v = V[i];

    // Convection. Backward (thesis): always the difference towards the west / south neighbour.
    // Upwind: the difference is taken on the side the flow comes from.
    const float du_x = upwind ? (u > 0.0f ? (u - n.uW) : (n.uE - u)) : (u - n.uW);
    const float dv_x = upwind ? (u > 0.0f ? (v - n.vW) : (n.vE - v)) : (v - n.vW);
    const float du_y = upwind ? (v > 0.0f ? (u - n.uS) : (n.uN - u)) : (u - n.uS);
    const float dv_y = upwind ? (v > 0.0f ? (v - n.vS) : (n.vN - v)) : (v - n.vS);
    const float Au = u * dt / dx * du_x;
    const float Av = u * dt / dx * dv_x;
    const float Bu = v * dt / dy * du_y;
    const float Bv = v * dt / dy * dv_y;
    const float Cu = (dt / (2.0f * rho * dx)) * (p.E - p.W);
    const float Cv = (dt / (2.0f * rho * dy)) * (p.N - p.S);
    const float Du = (dt / (dx * dx)) * (n.uE - 2.0f * u + n.uW);
    const float Dv = (dt / (dx * dx)) * (n.vE - 2.0f * v + n.vW);
    const float Eu = (dt / (dy * dy)) * (n.uN - 2.0f * u + n.uS);
    const float Ev = (dt / (dy * dy)) * (n.vN - 2.0f * v + n.vS);

    Uout[i] = u - Au - Bu - Cu + nu * (Du + Eu);
    Vout[i] = v - Av - Bv - Cv + nu * (Dv + Ev);
}
