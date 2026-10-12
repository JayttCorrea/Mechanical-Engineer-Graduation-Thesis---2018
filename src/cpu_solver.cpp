#include "tcc/cpu_solver.hpp"

#include <algorithm>

// Rows are independent inside every stage, so each stage can be split over threads.
#ifdef _OPENMP
#define TCC_PARALLEL_ROWS _Pragma("omp parallel for schedule(static)")
#else
#define TCC_PARALLEL_ROWS
#endif

namespace tcc {

namespace {

// Velocity of the four neighbours of cell (x, y), boundary conditions applied.
// Mirrors velocity_neighbours() in kernels/fluid.cl.
struct VelNb {
    float uN, vN, uS, vS, uE, vE, uW, vW;
};

inline VelNb velocity_neighbours(const Config& c, const float* U, const float* V, int x, int y) {
    const int nx = c.nx, ny = c.ny;
    const int i = y * nx + x;
    const bool tube = c.type == CaseType::Tube;
    VelNb n{};

    if (y == ny - 1) {
        n.uN = tube ? 0.0f : c.u0;
        n.vN = tube ? 0.0f : c.v0;
    } else {
        n.uN = U[i + nx];
        n.vN = V[i + nx];
    }

    if (y == 0) {
        n.uS = 0.0f;
        n.vS = 0.0f;
    } else {
        n.uS = U[i - nx];
        n.vS = V[i - nx];
    }

    if (x == nx - 1) {
        n.uE = tube ? U[i - 1] : 0.0f;  // free outflow copies the inner cell
        n.vE = tube ? V[i - 1] : 0.0f;
    } else {
        n.uE = U[i + 1];
        n.vE = V[i + 1];
    }

    if (x == 0) {
        n.uW = tube ? c.u0 : 0.0f;  // inlet
        n.vW = tube ? c.v0 : 0.0f;
    } else {
        n.uW = U[i - 1];
        n.vW = V[i - 1];
    }
    return n;
}

struct PresNb {
    float N, S, E, W;
};

// Zero normal gradient on every boundary. Mirrors pressure_neighbours() in fluid.cl.
inline PresNb pressure_neighbours(const Config& c, const float* P, int x, int y) {
    const int nx = c.nx, ny = c.ny;
    const int i = y * nx + x;
    const bool tube = c.type == CaseType::Tube;
    PresNb n{};
    n.N = (y == ny - 1) ? P[i - nx] : P[i + nx];
    n.S = (y == 0) ? P[i + nx] : P[i - nx];
    n.E = (x == nx - 1) ? P[i - 1] : P[i + 1];
    n.W = (x == 0) ? (tube ? P[i] : P[i + 1]) : P[i - 1];
    return n;
}

void compute_b(const Config& c, const float* U, const float* V, float* B) {
    const int nx = c.nx, ny = c.ny;
    const float dx = c.dx(), dy = c.dy(), rho = c.rho, dt = c.dt;
    TCC_PARALLEL_ROWS
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const VelNb n = velocity_neighbours(c, U, V, x, y);
            const float Ab = (n.uE - n.uW) / (2.0f * dx);
            const float Bb = (n.vN - n.vS) / (2.0f * dy);
            const float Cb = ((n.uE - n.uW) / (2.0f * dx)) * ((n.uE - n.uW) / (2.0f * dx));
            const float Db = 2.0f * ((n.uN - n.uS) / (2.0f * dy) * (n.vE - n.vW) / (2.0f * dx));
            const float Eb = ((n.vN - n.vS) / (2.0f * dy)) * ((n.vN - n.vS) / (2.0f * dy));
            const float Fp = (dx * dx) * (dy * dy) / (2.0f * ((dx * dx) + (dy * dy)));
            B[y * nx + x] = rho * Fp * ((1.0f / dt) * (Ab + Bb) - Cb - Db - Eb);
        }
    }
}

// One Jacobi sweep, Pin -> Pout (never the same buffer).
void pressure_iterate(const Config& c, const float* B, const float* Pin, float* Pout) {
    const int nx = c.nx, ny = c.ny;
    const float dx = c.dx(), dy = c.dy();
    TCC_PARALLEL_ROWS
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const PresNb p = pressure_neighbours(c, Pin, x, y);
            const float Ap = (p.E + p.W) * (dy * dy);
            const float Bp = (p.N + p.S) * (dx * dx);
            const float Cp = 2.0f * ((dx * dx) + (dy * dy));
            const int i = y * nx + x;
            Pout[i] = ((Ap + Bp) / Cp) - B[i];
        }
    }
}

void velocity_update(const Config& c, const float* U, const float* V, const float* P,
                     float* Uout, float* Vout) {
    const int nx = c.nx, ny = c.ny;
    const float dx = c.dx(), dy = c.dy(), rho = c.rho, nu = c.nu, dt = c.dt;
    TCC_PARALLEL_ROWS
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const int i = y * nx + x;
            const VelNb n = velocity_neighbours(c, U, V, x, y);
            const PresNb p = pressure_neighbours(c, P, x, y);
            const float u = U[i], v = V[i];

            float Au, Av, Bu, Bv;
            if (c.advection == Advection::Upwind) {
                Au = u * dt / dx * (u > 0.0f ? (u - n.uW) : (n.uE - u));
                Av = u * dt / dx * (u > 0.0f ? (v - n.vW) : (n.vE - v));
                Bu = v * dt / dy * (v > 0.0f ? (u - n.uS) : (n.uN - u));
                Bv = v * dt / dy * (v > 0.0f ? (v - n.vS) : (n.vN - v));
            } else {
                Au = u * dt / dx * (u - n.uW);
                Av = u * dt / dx * (v - n.vW);
                Bu = v * dt / dy * (u - n.uS);
                Bv = v * dt / dy * (v - n.vS);
            }
            const float Cu = (dt / (2.0f * rho * dx)) * (p.E - p.W);
            const float Cv = (dt / (2.0f * rho * dy)) * (p.N - p.S);
            const float Du = (dt / (dx * dx)) * (n.uE - 2.0f * u + n.uW);
            const float Dv = (dt / (dx * dx)) * (n.vE - 2.0f * v + n.vW);
            const float Eu = (dt / (dy * dy)) * (n.uN - 2.0f * u + n.uS);
            const float Ev = (dt / (dy * dy)) * (n.vN - 2.0f * v + n.vS);

            Uout[i] = u - Au - Bu - Cu + nu * (Du + Eu);
            Vout[i] = v - Av - Bv - Cv + nu * (Dv + Ev);
        }
    }
}

}  // namespace

CpuSolver::CpuSolver(const Config& cfg) : Solver(cfg) {
    const std::size_t n = cfg_.cells();
    fields_.assign(n);
    p_next_.assign(n, 0.0f);
    u_next_.assign(n, 0.0f);
    v_next_.assign(n, 0.0f);
}

std::string CpuSolver::backend_name() const {
#ifdef _OPENMP
    return "CPU (OpenMP)";
#else
    return "CPU (serial)";
#endif
}

void CpuSolver::reset() {
    fields_.assign(cfg_.cells());
    std::fill(p_next_.begin(), p_next_.end(), 0.0f);
    for (FieldGuard* g : {&guard_u_, &guard_v_, &guard_p_, &guard_b_}) g->clear();
    steps_ = 0;
    repaired_ = 0;
}

const Fields& CpuSolver::fields() {
    // The host arrays are the simulation state, so repairing them in place also lets the
    // run continue from valid numbers.
    repaired_ += guard_u_.repair(fields_.u) + guard_v_.repair(fields_.v) + guard_p_.repair(fields_.p) +
                 guard_b_.repair(fields_.b);
    return fields_;
}

void CpuSolver::set_velocity(const std::vector<float>& u, const std::vector<float>& v) {
    check_velocity_size(u, v);
    fields_.u = u;
    fields_.v = v;
}

void CpuSolver::step() {
    compute_b(cfg_, fields_.u.data(), fields_.v.data(), fields_.b.data());

    if (!cfg_.warm_start_pressure) std::fill(fields_.p.begin(), fields_.p.end(), 0.0f);
    for (int k = 0; k < cfg_.pressure_iterations; ++k) {
        pressure_iterate(cfg_, fields_.b.data(), fields_.p.data(), p_next_.data());
        fields_.p.swap(p_next_);
    }

    velocity_update(cfg_, fields_.u.data(), fields_.v.data(), fields_.p.data(),
                    u_next_.data(), v_next_.data());
    fields_.u.swap(u_next_);
    fields_.v.swap(v_next_);
    ++steps_;
}

}  // namespace tcc
