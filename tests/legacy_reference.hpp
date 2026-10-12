#pragma once
// Literal C++ transcription of the thesis kernels (Anexos 6-8 for the tube, Anexo 10
// for the cavity), including the original index decoding. It exists only so the tests
// can check the rewritten solver against the code that was published in 2017.
//
// Sources: tube   -> B_cl.cl / Pressure.cl / Velocity.cl (constant flow rate)
//          cavity -> CPU.cpp (lid on the top row, zero velocity on the other walls)

#include <algorithm>
#include <cmath>
#include <vector>

#include "tcc/config.hpp"

namespace legacy {

// Index decoding exactly as written in the thesis (integer division, then fix-ups).
inline void cell_xy(int i, int nx, int ny, int& X, int& Y) {
    float R = static_cast<float>(((i - 1) / nx) + 1);
    Y = static_cast<int>(std::ceil(R)) - 1;
    X = i - Y * nx;
    if (X == nx) X = 0;
    if (i == (nx * ny - 1)) {
        Y = ny - 1;
        X = nx - 1;
    }
    if (X == 0 && i != 0) Y++;
}

inline void b_cl(bool tube, const tcc::Config& c, const std::vector<float>& U,
                 const std::vector<float>& V, std::vector<float>& B) {
    const int nx = c.nx, ny = c.ny;
    const float dx = c.length_x / nx, dy = c.length_y / ny, rho = c.rho, dt = c.dt;
    for (int i = 0; i < nx * ny; ++i) {
        int X, Y;
        cell_xy(i, nx, ny, X, Y);
        float U_Sup, V_Sup, U_Inf, V_Inf, U_Direita, V_Direita, U_Esquerda, V_Esquerda;

        if (Y == ny - 1) {
            U_Sup = tube ? 0.0f : c.u0;
            V_Sup = tube ? 0.0f : c.v0;
        } else {
            U_Sup = U[i + nx];
            V_Sup = V[i + nx];
        }
        if (Y == 0) {
            U_Inf = 0.0f;
            V_Inf = 0.0f;
        } else {
            U_Inf = U[i - nx];
            V_Inf = V[i - nx];
        }
        if (X == nx - 1) {
            U_Direita = tube ? U[i - 1] : 0.0f;
            V_Direita = tube ? V[i - 1] : 0.0f;
        } else {
            U_Direita = U[i + 1];
            V_Direita = V[i + 1];
        }
        if (X == 0) {
            U_Esquerda = tube ? c.u0 : 0.0f;
            V_Esquerda = tube ? c.v0 : 0.0f;
        } else {
            U_Esquerda = U[i - 1];
            V_Esquerda = V[i - 1];
        }

        float Ab = (U_Direita - U_Esquerda) / (2.0f * dx);
        float Bb = (V_Sup - V_Inf) / (2.0f * dy);
        float Cb = ((U_Direita - U_Esquerda) / (2.0f * dx)) * ((U_Direita - U_Esquerda) / (2.0f * dx));
        float Db = 2.0f * ((U_Sup - U_Inf) / (2.0f * dy) * (V_Direita - V_Esquerda) / (2.0f * dx));
        float Eb = ((V_Sup - V_Inf) / (2.0f * dy)) * ((V_Sup - V_Inf) / (2.0f * dy));
        float Fp = (dx * dx) * (dy * dy) / (2.0f * ((dx * dx) + (dy * dy)));
        B[i] = rho * Fp * ((1.0f / dt) * (Ab + Bb) - Cb - Db - Eb);
    }
}

inline void pressure(bool tube, const tcc::Config& c, const std::vector<float>& B,
                     const std::vector<float>& Pin, std::vector<float>& Pout) {
    const int nx = c.nx, ny = c.ny;
    const float dx = c.length_x / nx, dy = c.length_y / ny;
    for (int i = 0; i < nx * ny; ++i) {
        int X, Y;
        cell_xy(i, nx, ny, X, Y);
        float P_Sup, P_Inf, P_Direita, P_Esquerda;

        P_Sup = (Y == ny - 1) ? Pin[i - nx] : Pin[i + nx];
        P_Inf = (Y == 0) ? Pin[i + nx] : Pin[i - nx];
        P_Direita = (X == nx - 1) ? Pin[i - 1] : Pin[i + 1];
        if (X == 0) P_Esquerda = tube ? Pin[i] : Pin[i + 1];
        else P_Esquerda = Pin[i - 1];

        float Ap = (P_Direita + P_Esquerda) * (dy * dy);
        float Bp = (P_Sup + P_Inf) * (dx * dx);
        float Cp = 2.0f * ((dx * dx) + (dy * dy));
        Pout[i] = ((Ap + Bp) / Cp) - B[i];
    }
}

inline void velocity(bool tube, const tcc::Config& c, const std::vector<float>& U,
                     const std::vector<float>& V, const std::vector<float>& P,
                     std::vector<float>& Uout, std::vector<float>& Vout) {
    const int nx = c.nx, ny = c.ny;
    const float dx = c.length_x / nx, dy = c.length_y / ny, rho = c.rho, dt = c.dt, nu = c.nu;
    for (int i = 0; i < nx * ny; ++i) {
        int X, Y;
        cell_xy(i, nx, ny, X, Y);
        float U_Sup, V_Sup, P_Sup, U_Inf, V_Inf, P_Inf;
        float U_Direita, V_Direita, P_Direita, U_Esquerda, V_Esquerda, P_Esquerda;

        if (Y == ny - 1) {
            U_Sup = tube ? 0.0f : c.u0;
            V_Sup = tube ? 0.0f : c.v0;
            P_Sup = P[i - nx];
        } else {
            U_Sup = U[i + nx];
            V_Sup = V[i + nx];
            P_Sup = P[i + nx];
        }
        if (Y == 0) {
            U_Inf = 0.0f;
            V_Inf = 0.0f;
            P_Inf = P[i + nx];
        } else {
            U_Inf = U[i - nx];
            V_Inf = V[i - nx];
            P_Inf = P[i - nx];
        }
        if (X == nx - 1) {
            U_Direita = tube ? U[i - 1] : 0.0f;
            V_Direita = tube ? V[i - 1] : 0.0f;
            P_Direita = P[i - 1];
        } else {
            U_Direita = U[i + 1];
            V_Direita = V[i + 1];
            P_Direita = P[i + 1];
        }
        if (X == 0) {
            U_Esquerda = tube ? c.u0 : 0.0f;
            V_Esquerda = tube ? c.v0 : 0.0f;
            P_Esquerda = tube ? P[i] : P[i + 1];
        } else {
            U_Esquerda = U[i - 1];
            V_Esquerda = V[i - 1];
            P_Esquerda = P[i - 1];
        }

        float Au = U[i] * dt / dx * (U[i] - U_Esquerda);
        float Av = U[i] * dt / dx * (V[i] - V_Esquerda);
        float Bu = V[i] * dt / dy * (U[i] - U_Inf);
        float Bv = V[i] * dt / dy * (V[i] - V_Inf);
        float Cu = (dt / (2.0f * rho * dx)) * (P_Direita - P_Esquerda);
        float Cv = (dt / (2.0f * rho * dy)) * (P_Sup - P_Inf);
        float Du = (dt / (dx * dx)) * (U_Direita - 2.0f * U[i] + U_Esquerda);
        float Dv = (dt / (dx * dx)) * (V_Direita - 2.0f * V[i] + V_Esquerda);
        float Eu = (dt / (dy * dy)) * (U_Sup - 2.0f * U[i] + U_Inf);
        float Ev = (dt / (dy * dy)) * (V_Sup - 2.0f * V[i] + V_Inf);

        Uout[i] = U[i] - Au - Bu - Cu + nu * (Du + Eu);
        Vout[i] = V[i] - Av - Bv - Cv + nu * (Dv + Ev);
    }
}

// One frame as the thesis Loop() runs it: pressure restarted from zero, B once,
// `ipf` Jacobi sweeps (with separate in/out buffers), velocity once.
struct State {
    std::vector<float> U, V, P, B;
    explicit State(std::size_t n) : U(n, 0.0f), V(n, 0.0f), P(n, 0.0f), B(n, 0.0f) {}
};

inline void step(bool tube, const tcc::Config& c, State& s, int ipf) {
    const std::size_t n = s.U.size();
    std::fill(s.B.begin(), s.B.end(), 0.0f);
    std::fill(s.P.begin(), s.P.end(), 0.0f);
    b_cl(tube, c, s.U, s.V, s.B);
    std::vector<float> tmp(n);
    for (int k = 0; k < ipf; ++k) {
        pressure(tube, c, s.B, s.P, tmp);
        s.P = tmp;
    }
    std::vector<float> un(n), vn(n);
    velocity(tube, c, s.U, s.V, s.P, un, vn);
    s.U = un;
    s.V = vn;
}

}  // namespace legacy
