// 2D incompressible Navier-Stokes, finite differences, one work-item per cell.
//
// Same maths as the thesis kernels (B_cl.cl, Pressure.cl, Velocity.cl) merged in a
// single program and launched on a 2D range instead of decoding a 1D id by hand.
//
// Layout : row-major, i = y * nx + x, y = 0 is the bottom row.
// Options: -DCASE_TUBE=1  inlet on the left (u0,v0), outflow on the right, no-slip walls
//          -DCASE_TUBE=0  lid-driven cavity: (u0,v0) is the lid velocity on the top row
//          -DUPWIND=0     thesis advection: backward difference whatever the sign of the velocity
//          -DUPWIND=1     first-order upwind (needed when the flow reverses, e.g. the cavity)

#ifndef CASE_TUBE
#define CASE_TUBE 1
#endif

#ifndef UPWIND
#define UPWIND 0
#endif

typedef struct { float uN, vN, uS, vS, uE, vE, uW, vW; } VelNb;

// Velocity of the four neighbours of cell (x, y) with boundary conditions applied.
inline VelNb velocity_neighbours(__global const float* restrict U,
                                 __global const float* restrict V,
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
inline PresNb pressure_neighbours(__global const float* restrict P,
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
__kernel void compute_b(__global float* restrict B,
                        __global const float* restrict U,
                        __global const float* restrict V,
                        const int nx, const int ny,
                        const float dx, const float dy,
                        const float rho, const float dt,
                        const float u0, const float v0)
{
    const int x = get_global_id(0), y = get_global_id(1);
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
__kernel void pressure_iterate(__global const float* restrict B,
                               __global const float* restrict Pin,
                               __global float* restrict Pout,
                               const int nx, const int ny,
                               const float dx, const float dy)
{
    const int x = get_global_id(0), y = get_global_id(1);
    const PresNb p = pressure_neighbours(Pin, x, y, nx, ny);

    const float Ap = (p.E + p.W) * (dy * dy);
    const float Bp = (p.N + p.S) * (dx * dx);
    const float Cp = 2.0f * ((dx * dx) + (dy * dy));
    const int i = y * nx + x;
    Pout[i] = ((Ap + Bp) / Cp) - B[i];
}

// Step 3: explicit momentum update (upwind advection, central pressure and diffusion).
__kernel void velocity_update(__global const float* restrict U,
                              __global const float* restrict V,
                              __global const float* restrict P,
                              __global float* restrict Uout,
                              __global float* restrict Vout,
                              const int nx, const int ny,
                              const float dx, const float dy,
                              const float rho, const float nu, const float dt,
                              const float u0, const float v0)
{
    const int x = get_global_id(0), y = get_global_id(1);
    const int i = y * nx + x;
    const VelNb n = velocity_neighbours(U, V, x, y, nx, ny, u0, v0);
    const PresNb p = pressure_neighbours(P, x, y, nx, ny);

    const float u = U[i], v = V[i];

#if UPWIND
    // difference taken on the side the flow comes from
    const float Au = u * dt / dx * (u > 0.0f ? (u - n.uW) : (n.uE - u));
    const float Av = u * dt / dx * (u > 0.0f ? (v - n.vW) : (n.vE - v));
    const float Bu = v * dt / dy * (v > 0.0f ? (u - n.uS) : (n.uN - u));
    const float Bv = v * dt / dy * (v > 0.0f ? (v - n.vS) : (n.vN - v));
#else
    const float Au = u * dt / dx * (u - n.uW);
    const float Av = u * dt / dx * (v - n.vW);
    const float Bu = v * dt / dy * (u - n.uS);
    const float Bv = v * dt / dy * (v - n.vS);
#endif
    const float Cu = (dt / (2.0f * rho * dx)) * (p.E - p.W);
    const float Cv = (dt / (2.0f * rho * dy)) * (p.N - p.S);
    const float Du = (dt / (dx * dx)) * (n.uE - 2.0f * u + n.uW);
    const float Dv = (dt / (dx * dx)) * (n.vE - 2.0f * v + n.vW);
    const float Eu = (dt / (dy * dy)) * (n.uN - 2.0f * u + n.uS);
    const float Ev = (dt / (dy * dy)) * (n.vN - 2.0f * v + n.vS);

    Uout[i] = u - Au - Bu - Cu + nu * (Du + Eu);
    Vout[i] = v - Av - Bv - Cv + nu * (Dv + Ev);
}
