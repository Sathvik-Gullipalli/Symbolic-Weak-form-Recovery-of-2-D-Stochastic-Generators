from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Library:
    names: list[str]
    preset: str
    funcs: tuple[Callable[[Array], Array], ...]

    def transform(self, x: Array) -> Array:
        x = np.asarray(x, float)
        if x.ndim == 1:
            x = x[:, None] if len(self.names) <= 4 else x[None, :]
        cols = [fn(x) for fn in self.funcs]
        return np.column_stack(cols).astype(float)


def _xy(x: Array) -> tuple[Array, Array]:
    if x.shape[1] == 1:
        return x[:, 0], np.zeros(x.shape[0])
    return x[:, 0], x[:, 1]


def _poly_terms(degree: int, state_names: tuple[str, str]) -> tuple[list[str], list[Callable[[Array], Array]]]:
    sx, sy = state_names
    names: list[str] = ["1"]
    funcs: list[Callable[[Array], Array]] = [lambda x: np.ones(x.shape[0])]
    for total in range(1, degree + 1):
        for px in range(total, -1, -1):
            py = total - px
            if py == 0:
                name = sx if px == 1 else f"{sx}^{px}"
            elif px == 0:
                name = sy if py == 1 else f"{sy}^{py}"
            else:
                left = sx if px == 1 else f"{sx}^{px}"
                right = sy if py == 1 else f"{sy}^{py}"
                name = f"{left}{right}"
            names.append(name)
            funcs.append(lambda x, px=px, py=py: (_xy(x)[0] ** px) * (_xy(x)[1] ** py))
    return names, funcs


def make_library(preset: str = "A", state_names: list[str] | tuple[str, str] = ("x", "y")) -> Library:
    preset = preset.upper()
    state_names = tuple(state_names)  # type: ignore[arg-type]
    if preset in {"A", "POLY2"}:
        names, funcs = _poly_terms(2, state_names)
    elif preset in {"B", "POLY3"}:
        names, funcs = _poly_terms(3, state_names)
    elif preset in {"C", "POLY4"}:
        names, funcs = _poly_terms(4, state_names)
    elif preset in {"D", "LOG_HESTON"}:
        sx, sy = state_names
        names = ["1", sx, sy, f"{sx}^2", f"{sx}{sy}", f"{sy}^2"]
        funcs = [
            lambda x: np.ones(x.shape[0]),
            lambda x: _xy(x)[0],
            lambda x: _xy(x)[1],
            lambda x: _xy(x)[0] ** 2,
            lambda x: _xy(x)[0] * _xy(x)[1],
            lambda x: _xy(x)[1] ** 2,
        ]
    elif preset in {"E", "HESTON_SV"}:
        sx, sy = state_names
        names = ["1", sx, sy, f"{sx}^2", f"{sx}{sy}", f"{sy}^2", f"{sx}^2{sy}", f"{sx}{sy}^2", f"{sx}^3", f"{sy}^3"]
        funcs = [
            lambda x: np.ones(x.shape[0]),
            lambda x: _xy(x)[0],
            lambda x: _xy(x)[1],
            lambda x: _xy(x)[0] ** 2,
            lambda x: _xy(x)[0] * _xy(x)[1],
            lambda x: _xy(x)[1] ** 2,
            lambda x: (_xy(x)[0] ** 2) * _xy(x)[1],
            lambda x: _xy(x)[0] * (_xy(x)[1] ** 2),
            lambda x: _xy(x)[0] ** 3,
            lambda x: _xy(x)[1] ** 3,
        ]
    elif preset in {"F", "SQRT_CIR"}:
        names = ["1", "x", "y", "sqrt(x)", "sqrt(y)", "sqrt(xy)", "xy", "x^2", "y^2"]
        funcs = [
            lambda x: np.ones(x.shape[0]),
            lambda x: _xy(x)[0],
            lambda x: _xy(x)[1],
            lambda x: np.sqrt(np.maximum(_xy(x)[0], 0.0)),
            lambda x: np.sqrt(np.maximum(_xy(x)[1], 0.0)),
            lambda x: np.sqrt(np.maximum(_xy(x)[0] * _xy(x)[1], 0.0)),
            lambda x: _xy(x)[0] * _xy(x)[1],
            lambda x: _xy(x)[0] ** 2,
            lambda x: _xy(x)[1] ** 2,
        ]
    elif preset in {"G", "TRIG", "POLY+TRIG", "POLY_TRIG"}:
        names, funcs = _poly_terms(2, state_names)
        names += ["sin x", "sin y", "cos x", "cos y"]
        funcs += [
            lambda x: np.sin(_xy(x)[0]),
            lambda x: np.sin(_xy(x)[1]),
            lambda x: np.cos(_xy(x)[0]),
            lambda x: np.cos(_xy(x)[1]),
        ]
    elif preset in {"POLY+RATIONAL", "POLY_RATIONAL", "RATIONAL"}:
        names, funcs = _poly_terms(2, state_names)
        names += ["1/(1+|x|)", "1/(1+|y|)", "x/(1+x^2)", "y/(1+y^2)"]
        funcs += [
            lambda x: 1.0 / (1.0 + np.abs(_xy(x)[0])),
            lambda x: 1.0 / (1.0 + np.abs(_xy(x)[1])),
            lambda x: _xy(x)[0] / (1.0 + _xy(x)[0] ** 2),
            lambda x: _xy(x)[1] / (1.0 + _xy(x)[1] ** 2),
        ]
    elif preset in {"POLY+RBF", "POLY_RBF", "RBF"}:
        names, funcs = _poly_terms(2, state_names)
        centers = [(-1.5, -1.5), (-1.5, 1.5), (0.0, 0.0), (1.5, -1.5), (1.5, 1.5)]
        for cx, cy in centers:
            names.append(f"rbf({cx:g},{cy:g})")
            funcs.append(lambda x, cx=cx, cy=cy: np.exp(-0.5 * ((_xy(x)[0] - cx) ** 2 + (_xy(x)[1] - cy) ** 2)))
    elif preset in {"FOURIER2", "FULL_FOURIER"}:
        sx, sy = state_names
        names = ["1", sx, sy]
        funcs = [lambda x: np.ones(x.shape[0]), lambda x: _xy(x)[0], lambda x: _xy(x)[1]]
        for freq in (1, 2):
            names += [f"sin{freq}{sx}", f"cos{freq}{sx}", f"sin{freq}{sy}", f"cos{freq}{sy}"]
            funcs += [
                lambda x, freq=freq: np.sin(freq * _xy(x)[0]),
                lambda x, freq=freq: np.cos(freq * _xy(x)[0]),
                lambda x, freq=freq: np.sin(freq * _xy(x)[1]),
                lambda x, freq=freq: np.cos(freq * _xy(x)[1]),
            ]
        names += ["sin x sin y", "cos x cos y"]
        funcs += [lambda x: np.sin(_xy(x)[0]) * np.sin(_xy(x)[1]), lambda x: np.cos(_xy(x)[0]) * np.cos(_xy(x)[1])]
    elif preset in {"LEGENDRE2", "CHEBYSHEV2", "HERMITE2"}:
        sx, sy = state_names
        fam = preset[:-1]
        names = ["1"]
        funcs = [lambda x: np.ones(x.shape[0])]

        def one(coord: int, degree: int, family: str) -> Callable[[Array], Array]:
            def fn(x: Array, coord: int = coord, degree: int = degree, family: str = family) -> Array:
                v = _xy(x)[coord]
                if family == "LEGENDRE":
                    if degree == 1:
                        return v
                    return 0.5 * (3.0 * v * v - 1.0)
                if family == "CHEBYSHEV":
                    if degree == 1:
                        return v
                    return 2.0 * v * v - 1.0
                if degree == 1:
                    return v
                return v * v - 1.0

            return fn

        for coord, label in enumerate((sx, sy)):
            names += [f"{fam.lower()}1({label})", f"{fam.lower()}2({label})"]
            funcs += [one(coord, 1, fam), one(coord, 2, fam)]
        names += [f"{fam.lower()}1({sx}){fam.lower()}1({sy})"]
        funcs += [lambda x, fam=fam: one(0, 1, fam)(x) * one(1, 1, fam)(x)]
    else:
        raise ValueError(f"unknown library preset {preset!r}")
    return Library(names=list(names), preset=preset, funcs=tuple(funcs))


def polynomial_change_of_basis(mean: Array, scale: Array, degree: int) -> Array:
    """Map degree-graded z-monomial coefficients to raw x-monomial coefficients."""
    powers: list[tuple[int, int]] = []
    for total in range(degree + 1):
        for px in range(total, -1, -1):
            powers.append((px, total - px))
    idx = {p: i for i, p in enumerate(powers)}
    m = np.zeros((len(powers), len(powers)))
    mu1, mu2 = np.asarray(mean, float)[:2]
    s1, s2 = np.asarray(scale, float)[:2]
    for col, (a, b) in enumerate(powers):
        for i in range(a + 1):
            for k in range(b + 1):
                coef = (s1 ** -a) * (s2 ** -b) * comb(a, i) * comb(b, k)
                coef *= ((-mu1) ** (a - i)) * ((-mu2) ** (b - k))
                m[idx[(i, k)], col] += coef
    return m


def make_heston_minimal_library() -> Library:
    return make_library("D", ("X", "V"))
