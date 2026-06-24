from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Type

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

from .simulate import euler_maruyama_correlated

Array = np.ndarray


class System:
    dim = 2
    name = "system"
    library_hint = "A"

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.zeros(self.dim) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed)

    def true_drift(self, x: Array) -> Array:
        raise NotImplementedError

    def true_diffusion(self, x: Array) -> Array:
        raise NotImplementedError

    def true_current(self, x: Array) -> Array | None:
        return None

    def true_eigenvalues(self, n: int) -> Array | None:
        return None


def _atleast(x: Array) -> Array:
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None, :]
    return x


class OU(System):
    dim = 1
    name = "ou"
    library_hint = "A"

    def __init__(self, alpha: float = 1.0, sigma: float = 1.0):
        self.alpha, self.sigma = alpha, sigma

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 1))
        out[0, 0] = 0.0 if X0 is None else float(np.asarray(X0).ravel()[0])
        decay = np.exp(-self.alpha * dt)
        vol = self.sigma * np.sqrt((1.0 - np.exp(-2 * self.alpha * dt)) / (2 * self.alpha))
        for n in range(M):
            out[n + 1, 0] = decay * out[n, 0] + vol * rng.standard_normal()
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        return -self.alpha * x[:, :1]

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.full((x.shape[0], 1, 1), self.sigma**2)

    def true_eigenvalues(self, n: int) -> Array:
        return -self.alpha * np.arange(1, n + 1)


class DoubleWell(OU):
    name = "double_well"
    library_hint = "B"

    def __init__(self, sigma: float = 0.7):
        self.sigma = sigma
        self.alpha = 1.0

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 1))
        out[0, 0] = -1.0 if X0 is None else float(np.asarray(X0).ravel()[0])
        for n in range(M):
            x = out[n, 0]
            out[n + 1, 0] = np.clip(x + (x - x**3) * dt + self.sigma * np.sqrt(dt) * rng.standard_normal(), -4, 4)
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        return x[:, :1] - x[:, :1] ** 3

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.full((x.shape[0], 1, 1), self.sigma**2)


class MultiplicativeDiffusion(OU):
    name = "multiplicative"
    library_hint = "A"

    def __init__(self, sigma0: float = 1.0):
        self.sigma0 = sigma0
        self.alpha = 1.0
        self.sigma = sigma0

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 1))
        out[0, 0] = 0.0 if X0 is None else float(np.asarray(X0).ravel()[0])
        for n in range(M):
            x = out[n, 0]
            out[n + 1, 0] = np.clip(x - x * dt + self.sigma0 * np.sqrt(1 + x * x) * np.sqrt(dt) * rng.standard_normal(), -8, 8)
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        return -x[:, :1]

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return (self.sigma0**2 * (1 + x[:, :1] ** 2))[:, :, None]


class LinearOU2D(System):
    name = "linear_ou_2d"
    library_hint = "A"

    def __init__(self, A: Array, diffusion: Array):
        self.A = np.asarray(A, float)
        self.diffusion = np.asarray(diffusion, float)

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        x0 = np.zeros(2) if X0 is None else np.asarray(X0, float)
        phi = expm(self.A * dt)
        sigma_inf = solve_continuous_lyapunov(self.A, -self.diffusion)
        qdt = sigma_inf - phi @ sigma_inf @ phi.T
        vals, vecs = np.linalg.eigh(0.5 * (qdt + qdt.T))
        root = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-14)))
        out = np.zeros((M + 1, 2))
        out[0] = x0
        for n in range(M):
            out[n + 1] = phi @ out[n] + root @ rng.standard_normal(2)
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        return x @ self.A.T

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat(self.diffusion[None, :, :], x.shape[0], axis=0)

    def true_eigenvalues(self, n: int) -> Array:
        return np.linalg.eigvals(self.A)[:n]


class IndependentOU2D(LinearOU2D):
    name = "indep_ou"

    def __init__(self, theta1: float = 1.0, theta2: float = 2.0, sigma1: float = 1.0, sigma2: float = 0.7):
        super().__init__([[-theta1, 0.0], [0.0, -theta2]], [[sigma1**2, 0.0], [0.0, sigma2**2]])


class CorrelatedOU2D(LinearOU2D):
    name = "correlated_ou"

    def __init__(self, theta1: float = 1.0, theta2: float = 1.5, sigma1: float = 1.0, sigma2: float = 0.8, rho: float = -0.6):
        self.rho = rho
        super().__init__([[-theta1, 0.0], [0.0, -theta2]], [[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]])


class CoupledLinearOU(LinearOU2D):
    name = "coupled_ou"

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.5, d: float = 0.5, sigma: float = 1.0):
        super().__init__([[-a, c], [d, -b]], [[sigma**2, 0.0], [0.0, sigma**2]])


class RotationalOU(LinearOU2D):
    name = "rotational_ou"

    def __init__(self, alpha: float = 1.0, omega: float = 2.0, sigma: float = 1.0):
        self.alpha, self.omega, self.sigma = alpha, omega, sigma
        super().__init__([[-alpha, -omega], [omega, -alpha]], [[sigma**2, 0.0], [0.0, sigma**2]])

    def true_current(self, x: Array) -> Array:
        x = _atleast(x)
        return self.omega * np.column_stack([-x[:, 1], x[:, 0]])


class SpiralSinkCorrelated(LinearOU2D):
    name = "spiral_sink_corr"

    def __init__(self, gamma: float = 1.0, omega: float = 1.5, sigma1: float = 1.0, sigma2: float = 0.8, rho: float = -0.5):
        super().__init__([[-gamma, -omega], [omega, -gamma]], [[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]])

    def true_current(self, x: Array) -> Array:
        x = _atleast(x)
        sigma_inf = solve_continuous_lyapunov(self.A, -self.diffusion)
        omega = self.A @ sigma_inf + 0.5 * self.diffusion
        mat = omega @ np.linalg.pinv(sigma_inf)
        return x @ mat.T


class DoubleWellTransverse(System):
    name = "double_well_transverse"
    library_hint = "B"

    def __init__(self, beta: float = 0.5, lam: float = 1.0, sigma: float = 0.7):
        self.beta, self.lam, self.sigma = beta, lam, sigma

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.array([-1.0, 0.0]) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed, clip=(-5.0, 5.0))

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        return np.column_stack([xx - xx**3 - self.beta * yy, -self.lam * yy + self.beta * xx])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat((self.sigma**2 * np.eye(2))[None, :, :], x.shape[0], axis=0)


class GradientPotential2D(DoubleWellTransverse):
    name = "gradient_potential"
    library_hint = "B"

    def __init__(self, lam: float = 1.0, eta: float = 0.25, sigma: float = 0.7):
        self.lam, self.eta, self.sigma = lam, eta, sigma
        self.beta = 0.0

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        return np.column_stack([xx - xx**3 - 2 * self.eta * xx * yy**2, -self.lam * yy - 2 * self.eta * xx**2 * yy])


class NonGradientDoubleWell(GradientPotential2D):
    name = "nongradient_circulation"

    def __init__(self, omega: float = 1.0, lam: float = 1.0, eta: float = 0.25, sigma: float = 0.7):
        super().__init__(lam, eta, sigma)
        self.omega = omega

    def true_drift(self, x: Array) -> Array:
        grad_neg = super().true_drift(x)
        grad = -grad_neg
        rot = self.omega * np.column_stack([-grad[:, 1], grad[:, 0]])
        return grad_neg + rot

    def true_current(self, x: Array) -> Array:
        return self.true_drift(x) - GradientPotential2D.true_drift(self, x)


class DiagonalMultiplicative2D(System):
    name = "diag_multiplicative"
    library_hint = "A"

    def __init__(self, a0: float = 0.5, ax: float = 0.1, ay: float = 0.1, b0: float = 0.4, bx: float = 0.1, by: float = 0.1, theta: float = 1.0):
        self.a0, self.ax, self.ay, self.b0, self.bx, self.by, self.theta = a0, ax, ay, b0, bx, by, theta

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.zeros(2) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed, clip=(-6.0, 6.0))

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        return -self.theta * x

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        a = np.zeros((x.shape[0], 2, 2))
        a[:, 0, 0] = self.a0 + self.ax * xx**2 + self.ay * yy**2
        a[:, 1, 1] = self.b0 + self.bx * xx**2 + self.by * yy**2
        return a


class CholeskyDiffusion2D(DiagonalMultiplicative2D):
    name = "nondiag_cholesky"
    library_hint = "C"

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        l11 = 0.5 + 0.1 * xx**2
        l21 = 0.2 * xx * yy
        l22 = 0.4 + 0.1 * yy**2
        a = np.zeros((x.shape[0], 2, 2))
        a[:, 0, 0] = l11**2
        a[:, 0, 1] = a[:, 1, 0] = l11 * l21
        a[:, 1, 1] = l21**2 + l22**2
        return a


class NearSingularDiffusion2D(DiagonalMultiplicative2D):
    name = "near_singular"

    def __init__(self, ratio: float = 0.95, theta: float = 1.0):
        super().__init__(theta=theta)
        self.ratio = ratio

    def true_diffusion(self, x: Array) -> Array:
        a = super().true_diffusion(x)
        a[:, 0, 1] = a[:, 1, 0] = self.ratio * np.sqrt(a[:, 0, 0] * a[:, 1, 1])
        return a


class LogHestonSV(System):
    name = "heston_logsv"
    library_hint = "D"

    def __init__(self, mu: float = 0.05, kappa: float = 2.0, theta: float = 0.04, xi: float = 0.3, rho: float = -0.65, floor: float = 1e-10):
        self.mu, self.kappa, self.theta, self.xi, self.rho, self.floor = mu, kappa, theta, xi, rho, floor

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 2))
        out[0] = np.array([0.0, self.theta]) if X0 is None else np.asarray(X0, float)
        chol = np.linalg.cholesky(np.array([[1.0, self.rho], [self.rho, 1.0]]))
        for n in range(M):
            x, v = out[n]
            vp = max(v, self.floor)
            z = chol @ rng.standard_normal(2)
            out[n + 1, 0] = x + (self.mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * z[0]
            out[n + 1, 1] = max(v + self.kappa * (self.theta - vp) * dt + self.xi * np.sqrt(vp * dt) * z[1], self.floor)
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        v = np.maximum(x[:, 1], self.floor)
        return np.column_stack([self.mu - 0.5 * v, self.kappa * (self.theta - v)])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        v = np.maximum(x[:, 1], self.floor)
        a = np.zeros((x.shape[0], 2, 2))
        a[:, 0, 0] = v
        a[:, 1, 1] = self.xi**2 * v
        a[:, 0, 1] = a[:, 1, 0] = self.rho * self.xi * v
        return a


class HestonSV(LogHestonSV):
    name = "heston_sv"
    library_hint = "E"

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 2))
        out[0] = np.array([100.0, self.theta]) if X0 is None else np.asarray(X0, float)
        chol = np.linalg.cholesky(np.array([[1.0, self.rho], [self.rho, 1.0]]))
        for n in range(M):
            s, v = out[n]
            vp = max(v, self.floor)
            z = chol @ rng.standard_normal(2)
            s_next = s + self.mu * s * dt + s * np.sqrt(vp * dt) * z[0]
            v_next = v + self.kappa * (self.theta - vp) * dt + self.xi * np.sqrt(vp * dt) * z[1]
            out[n + 1] = [max(s_next, self.floor), max(v_next, self.floor)]
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        s, v = x[:, 0], np.maximum(x[:, 1], self.floor)
        return np.column_stack([self.mu * s, self.kappa * (self.theta - v)])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        s, v = x[:, 0], np.maximum(x[:, 1], self.floor)
        a = np.zeros((x.shape[0], 2, 2))
        a[:, 0, 0] = s * s * v
        a[:, 1, 1] = self.xi**2 * v
        a[:, 0, 1] = a[:, 1, 0] = self.rho * self.xi * s * v
        return a


class CIRPairCorrelated(LogHestonSV):
    name = "cir_pair"
    library_hint = "F"

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        return np.column_stack([self.kappa * (self.theta - x[:, 0]), self.kappa * (self.theta - x[:, 1])])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        xx = np.maximum(x[:, 0], self.floor)
        yy = np.maximum(x[:, 1], self.floor)
        a = np.zeros((x.shape[0], 2, 2))
        a[:, 0, 0] = self.xi**2 * xx
        a[:, 1, 1] = self.xi**2 * yy
        a[:, 0, 1] = a[:, 1, 0] = self.rho * self.xi**2 * np.sqrt(xx * yy)
        return a

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 2))
        out[0] = np.array([self.theta, self.theta]) if X0 is None else np.asarray(X0, float)
        chol = np.linalg.cholesky(np.array([[1.0, self.rho], [self.rho, 1.0]]))
        for n in range(M):
            x = np.maximum(out[n], self.floor)
            z = chol @ rng.standard_normal(2)
            nxt = out[n] + self.kappa * (self.theta - x) * dt + self.xi * np.sqrt(x * dt) * z
            out[n + 1] = np.maximum(nxt, self.floor)
        return out


class UnderdampedLangevin(System):
    name = "underdamped_langevin"
    library_hint = "B"

    def __init__(self, gamma: float = 1.0, sigma: float = 1.0):
        self.gamma, self.sigma = gamma, sigma

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 2))
        out[0] = np.array([-1.0, 0.0]) if X0 is None else np.asarray(X0, float)
        for n in range(M):
            q, p = out[n]
            z1, z2 = rng.standard_normal(2)
            p_half = p - 0.5 * dt * ((q**3 - q) + self.gamma * p) + 0.5 * self.sigma * np.sqrt(dt) * z1
            q_next = q + dt * p_half
            p_next = p_half - 0.5 * dt * ((q_next**3 - q_next) + self.gamma * p_half) + 0.5 * self.sigma * np.sqrt(dt) * z2
            out[n + 1] = np.clip([q_next, p_next], -6.0, 6.0)
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        q, p = x[:, 0], x[:, 1]
        return np.column_stack([p, -self.gamma * p - (q**3 - q)])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        a = np.zeros((x.shape[0], 2, 2))
        a[:, 1, 1] = self.sigma**2
        return a


class NonPolynomialDrift2D(System):
    name = "nonpoly_drift"
    library_hint = "G"

    def __init__(self, sigma: float = 0.7):
        self.sigma = sigma

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.zeros(2) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed, clip=(-6.0, 6.0))

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        return np.column_stack([-x[:, 0] + np.sin(x[:, 1]), -x[:, 1] + np.cos(x[:, 0])])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat((self.sigma**2 * np.eye(2))[None, :, :], x.shape[0], axis=0)


class NearBoundaryHeston(LogHestonSV):
    name = "near_boundary_heston"
    library_hint = "D"

    def __init__(self):
        super().__init__(mu=0.05, kappa=1.0, theta=0.02, xi=0.4, rho=-0.65, floor=1e-10)


class PartialObservationRotOU(OU):
    name = "partial_observation"
    library_hint = "A"

    def __init__(self):
        super().__init__(alpha=1.0, sigma=1.0)
        self.full_system = RotationalOU()

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        full_x0 = None if X0 is None else np.array([float(np.asarray(X0).ravel()[0]), 0.0])
        return self.full_system.simulate(full_x0, dt, M, seed)[:, :1]


class BadCoverageRotOU(RotationalOU):
    name = "bad_coverage"
    library_hint = "A"

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        x0 = np.array([2.0, 2.0]) + 0.03 * rng.standard_normal(2) if X0 is None else np.asarray(X0, float)
        short_m = min(M, max(200, M // 8))
        path = super().simulate(x0, dt, short_m, seed)
        if short_m == M:
            return path
        return np.vstack([path, np.repeat(path[-1:], M - short_m, axis=0)])


class TooLargeDtMultiplicative(DiagonalMultiplicative2D):
    name = "too_large_dt"
    library_hint = "A"


class VanDerPolOscillator(System):
    name = "van_der_pol"
    library_hint = "B"

    def __init__(self, mu: float = 1.2, sigma: float = 0.35):
        self.mu, self.sigma = mu, sigma

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.array([1.0, 0.0]) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed, clip=(-4.0, 4.0))

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        return np.column_stack([yy, self.mu * (1.0 - xx**2) * yy - xx])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat((self.sigma**2 * np.eye(2))[None, :, :], x.shape[0], axis=0)


class FitzHughNagumo(System):
    name = "fitzhugh_nagumo"
    library_hint = "B"

    def __init__(self, eps: float = 0.08, a: float = 0.7, b: float = 0.8, current: float = 0.5, sigma: float = 0.25):
        self.eps, self.a, self.b, self.current, self.sigma = eps, a, b, current, sigma

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.array([-1.0, 1.0]) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed, clip=(-4.0, 4.0))

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        return np.column_stack([xx - xx**3 / 3.0 - yy + self.current, self.eps * (xx + self.a - self.b * yy)])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat((self.sigma**2 * np.eye(2))[None, :, :], x.shape[0], axis=0)


class StuartLandauOscillator(System):
    name = "stuart_landau"
    library_hint = "B"

    def __init__(self, lam: float = 1.0, omega: float = 1.5, sigma: float = 0.28):
        self.lam, self.omega, self.sigma = lam, omega, sigma

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.array([1.0, 0.0]) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed, clip=(-3.5, 3.5))

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        r2 = xx * xx + yy * yy
        return np.column_stack([(self.lam - r2) * xx - self.omega * yy, self.omega * xx + (self.lam - r2) * yy])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat((self.sigma**2 * np.eye(2))[None, :, :], x.shape[0], axis=0)

    def true_current(self, x: Array) -> Array:
        x = _atleast(x)
        return self.omega * np.column_stack([-x[:, 1], x[:, 0]])


class Brusselator(System):
    name = "brusselator"
    library_hint = "B"

    def __init__(self, A: float = 1.0, B: float = 2.6, sigma: float = 0.12, floor: float = 1e-8):
        self.A, self.B, self.sigma, self.floor = A, B, sigma, floor

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 2))
        out[0] = np.array([self.A, self.B / self.A]) if X0 is None else np.asarray(X0, float)
        root = self.sigma * math.sqrt(dt)
        for n in range(M):
            nxt = out[n] + self.true_drift(out[n : n + 1])[0] * dt + root * rng.standard_normal(2)
            out[n + 1] = np.clip(nxt, self.floor, 6.0)
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = np.maximum(x[:, 0], self.floor), np.maximum(x[:, 1], self.floor)
        return np.column_stack([self.A - (self.B + 1.0) * xx + xx * xx * yy, self.B * xx - xx * xx * yy])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat((self.sigma**2 * np.eye(2))[None, :, :], x.shape[0], axis=0)


class MaierStein(System):
    name = "maier_stein"
    library_hint = "B"

    def __init__(self, beta: float = 0.35, sigma: float = 0.35):
        self.beta, self.sigma = beta, sigma

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.array([-1.0, 0.0]) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed, clip=(-4.0, 4.0))

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        return np.column_stack([xx - xx**3 - self.beta * xx * yy**2, -(1.0 + xx**2) * yy])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat((self.sigma**2 * np.eye(2))[None, :, :], x.shape[0], axis=0)


class DuffingOscillator(System):
    name = "duffing"
    library_hint = "B"

    def __init__(self, delta: float = 0.35, sigma: float = 0.30):
        self.delta, self.sigma = delta, sigma

    def simulate(self, X0: Array | None = None, dt: float = 0.01, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.array([-1.0, 0.0]) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed, clip=(-4.0, 4.0))

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        return np.column_stack([yy, -self.delta * yy + xx - xx**3])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat((self.sigma**2 * np.eye(2))[None, :, :], x.shape[0], axis=0)


class MuellerBrownPotential(System):
    name = "mueller_brown"
    library_hint = "C"

    def __init__(self, mobility: float = 0.004, sigma: float = 0.35):
        self.mobility, self.sigma = mobility, sigma
        self.A = np.array([-200.0, -100.0, -170.0, 15.0])
        self.a = np.array([-1.0, -1.0, -6.5, 0.7])
        self.b = np.array([0.0, 0.0, 11.0, 0.6])
        self.c = np.array([-10.0, -10.0, -6.5, 0.7])
        self.x0 = np.array([1.0, 0.0, -0.5, -1.0])
        self.y0 = np.array([0.0, 0.5, 1.5, 1.0])

    def simulate(self, X0: Array | None = None, dt: float = 0.002, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 2))
        out[0] = np.array([-0.55, 1.45]) if X0 is None else np.asarray(X0, float)
        root = self.sigma * math.sqrt(dt)
        for n in range(M):
            b = np.clip(self.true_drift(out[n : n + 1])[0], -25.0, 25.0)
            nxt = out[n] + b * dt + root * rng.standard_normal(2)
            out[n + 1] = np.array([np.clip(nxt[0], -1.7, 1.3), np.clip(nxt[1], -0.4, 2.1)])
        return out

    def _grad_potential(self, x: Array) -> tuple[Array, Array]:
        xx, yy = x[:, 0], x[:, 1]
        gx = np.zeros(x.shape[0])
        gy = np.zeros(x.shape[0])
        for A, a, b, c, x0, y0 in zip(self.A, self.a, self.b, self.c, self.x0, self.y0):
            dx = xx - x0
            dy = yy - y0
            expo = np.clip(a * dx * dx + b * dx * dy + c * dy * dy, -80.0, 80.0)
            term = A * np.exp(expo)
            gx += term * (2.0 * a * dx + b * dy)
            gy += term * (b * dx + 2.0 * c * dy)
        return gx, gy

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        gx, gy = self._grad_potential(x)
        return -self.mobility * np.column_stack([gx, gy])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        return np.repeat((self.sigma**2 * np.eye(2))[None, :, :], x.shape[0], axis=0)


class SABR(System):
    name = "sabr"
    library_hint = "F"

    def __init__(self, beta: float = 0.5, nu: float = 0.45, rho: float = -0.55, floor: float = 1e-8):
        self.beta, self.nu, self.rho, self.floor = beta, nu, rho, floor

    def simulate(self, X0: Array | None = None, dt: float = 1.0 / 252.0, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 2))
        out[0] = np.array([1.0, 0.25]) if X0 is None else np.asarray(X0, float)
        chol = np.linalg.cholesky(np.array([[1.0, self.rho], [self.rho, 1.0]]))
        for n in range(M):
            f, vol = np.maximum(out[n], self.floor)
            z = chol @ rng.standard_normal(2)
            f_next = f + vol * f**self.beta * math.sqrt(dt) * z[0]
            vol_next = vol + self.nu * vol * math.sqrt(dt) * z[1]
            out[n + 1] = [np.clip(f_next, self.floor, 5.0), np.clip(vol_next, self.floor, 2.0)]
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        return np.zeros((x.shape[0], 2))

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        f = np.maximum(x[:, 0], self.floor)
        vol = np.maximum(x[:, 1], self.floor)
        f_beta = f**self.beta
        a = np.zeros((x.shape[0], 2, 2))
        a[:, 0, 0] = vol**2 * f ** (2.0 * self.beta)
        a[:, 1, 1] = self.nu**2 * vol**2
        a[:, 0, 1] = a[:, 1, 0] = self.rho * self.nu * vol**2 * f_beta
        return a


class CorrelatedGBM2D(System):
    name = "gbm_2d"
    library_hint = "A"

    def __init__(self, mu1: float = 0.04, mu2: float = 0.02, sigma1: float = 0.22, sigma2: float = 0.30, rho: float = 0.45, floor: float = 1e-8):
        self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.floor = mu1, mu2, sigma1, sigma2, rho, floor

    def simulate(self, X0: Array | None = None, dt: float = 1.0 / 252.0, M: int = 1000, seed: int | None = None) -> Array:
        rng = np.random.default_rng(seed)
        out = np.zeros((M + 1, 2))
        out[0] = np.array([1.0, 1.2]) if X0 is None else np.asarray(X0, float)
        chol = np.linalg.cholesky(np.array([[1.0, self.rho], [self.rho, 1.0]]))
        drift = np.array([self.mu1 - 0.5 * self.sigma1**2, self.mu2 - 0.5 * self.sigma2**2]) * dt
        scale = np.array([self.sigma1, self.sigma2]) * math.sqrt(dt)
        for n in range(M):
            z = chol @ rng.standard_normal(2)
            out[n + 1] = np.maximum(out[n] * np.exp(drift + scale * z), self.floor)
        return out

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        return np.column_stack([self.mu1 * x[:, 0], self.mu2 * x[:, 1]])

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        s1 = np.maximum(x[:, 0], self.floor)
        s2 = np.maximum(x[:, 1], self.floor)
        a = np.zeros((x.shape[0], 2, 2))
        a[:, 0, 0] = self.sigma1**2 * s1**2
        a[:, 1, 1] = self.sigma2**2 * s2**2
        a[:, 0, 1] = a[:, 1, 0] = self.rho * self.sigma1 * self.sigma2 * s1 * s2
        return a


class TwoFactorVasicek(System):
    name = "two_factor_vasicek"
    library_hint = "A"

    def __init__(self, kappa1: float = 1.1, kappa2: float = 0.7, theta1: float = 0.03, theta2: float = 0.05, coupling12: float = 0.25, coupling21: float = -0.15, sigma1: float = 0.018, sigma2: float = 0.024, rho: float = 0.35):
        self.kappa1, self.kappa2, self.theta1, self.theta2 = kappa1, kappa2, theta1, theta2
        self.coupling12, self.coupling21 = coupling12, coupling21
        self.sigma1, self.sigma2, self.rho = sigma1, sigma2, rho

    def simulate(self, X0: Array | None = None, dt: float = 1.0 / 252.0, M: int = 1000, seed: int | None = None) -> Array:
        x0 = np.array([self.theta1, self.theta2]) if X0 is None else np.asarray(X0, float)
        return euler_maruyama_correlated(self.true_drift, self.true_diffusion, x0, dt, M, seed, clip=(-0.12, 0.18))

    def true_drift(self, x: Array) -> Array:
        x = _atleast(x)
        xx, yy = x[:, 0], x[:, 1]
        return np.column_stack(
            [
                self.kappa1 * (self.theta1 - xx) + self.coupling12 * (yy - self.theta2),
                self.kappa2 * (self.theta2 - yy) + self.coupling21 * (xx - self.theta1),
            ]
        )

    def true_diffusion(self, x: Array) -> Array:
        x = _atleast(x)
        a = np.array([[self.sigma1**2, self.rho * self.sigma1 * self.sigma2], [self.rho * self.sigma1 * self.sigma2, self.sigma2**2]])
        return np.repeat(a[None, :, :], x.shape[0], axis=0)


@dataclass(frozen=True)
class SystemTruth:
    cls: Type[System]
    dim: int
    tier: str
    verdict: str
    library: str


REGISTRY: dict[str, SystemTruth] = {
    "ou": SystemTruth(OU, 1, "0", "STRONG", "A"),
    "double_well": SystemTruth(DoubleWell, 1, "0", "STRONG", "B"),
    "multiplicative": SystemTruth(MultiplicativeDiffusion, 1, "0", "STRONG", "A"),
    "indep_ou": SystemTruth(IndependentOU2D, 2, "1", "STRONG", "A"),
    "correlated_ou": SystemTruth(CorrelatedOU2D, 2, "1", "STRONG", "A"),
    "coupled_ou": SystemTruth(CoupledLinearOU, 2, "1", "STRONG", "A"),
    "rotational_ou": SystemTruth(RotationalOU, 2, "2", "STRONG", "A"),
    "spiral_sink_corr": SystemTruth(SpiralSinkCorrelated, 2, "2", "STRONG", "A"),
    "double_well_transverse": SystemTruth(DoubleWellTransverse, 2, "3", "STRONG", "B"),
    "gradient_potential": SystemTruth(GradientPotential2D, 2, "3", "STRONG", "B"),
    "nongradient_circulation": SystemTruth(NonGradientDoubleWell, 2, "3", "STRONG", "B"),
    "diag_multiplicative": SystemTruth(DiagonalMultiplicative2D, 2, "4", "STRONG", "A"),
    "nondiag_cholesky": SystemTruth(CholeskyDiffusion2D, 2, "4", "STRONG", "C"),
    "near_singular": SystemTruth(NearSingularDiffusion2D, 2, "4", "FRAGILE", "A"),
    "heston_sv": SystemTruth(HestonSV, 2, "5", "FRAGILE", "E"),
    "heston_logsv": SystemTruth(LogHestonSV, 2, "5", "STRONG", "D"),
    "cir_pair": SystemTruth(CIRPairCorrelated, 2, "5", "FRAGILE", "F"),
    "underdamped_langevin": SystemTruth(UnderdampedLangevin, 2, "6", "FRAGILE", "B"),
    "near_boundary_heston": SystemTruth(NearBoundaryHeston, 2, "6", "FRAGILE", "D"),
    "nonpoly_drift": SystemTruth(NonPolynomialDrift2D, 2, "6", "STRONG_WITH_G_FAIL_WITHOUT", "G"),
    "partial_observation": SystemTruth(PartialObservationRotOU, 1, "6", "FRAGILE_FAIL", "A"),
    "bad_coverage": SystemTruth(BadCoverageRotOU, 2, "6", "FRAGILE_FAIL", "A"),
    "too_large_dt": SystemTruth(TooLargeDtMultiplicative, 2, "6", "FRAGILE_FAIL", "A"),
    "van_der_pol": SystemTruth(VanDerPolOscillator, 2, "7-limit-cycle", "STRONG", "B"),
    "fitzhugh_nagumo": SystemTruth(FitzHughNagumo, 2, "7-limit-cycle", "STRONG", "B"),
    "stuart_landau": SystemTruth(StuartLandauOscillator, 2, "7-limit-cycle", "STRONG", "B"),
    "brusselator": SystemTruth(Brusselator, 2, "7-limit-cycle", "STRONG", "B"),
    "maier_stein": SystemTruth(MaierStein, 2, "8-bistable", "STRONG", "B"),
    "duffing": SystemTruth(DuffingOscillator, 2, "8-bistable", "STRONG", "B"),
    "mueller_brown": SystemTruth(MuellerBrownPotential, 2, "8-bistable", "FRAGILE_FAIL", "C"),
    "sabr": SystemTruth(SABR, 2, "9-financial", "FRAGILE", "F"),
    "gbm_2d": SystemTruth(CorrelatedGBM2D, 2, "9-financial", "STRONG", "A"),
    "two_factor_vasicek": SystemTruth(TwoFactorVasicek, 2, "9-financial", "STRONG", "A"),
}
