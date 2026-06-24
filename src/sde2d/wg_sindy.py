from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .generator import GeneratorFit2D, fit_generator_2d
from .library import Library


@dataclass(frozen=True)
class WGSINDyDefaults:
    """Frozen v6 operating point for WG-SINDy."""

    center_scheme: str = "kmeans"
    n_centers: int = 64
    bandwidth_multiplier: float = 1.5
    bandwidth_rule: str = "cov"
    knn_k: int = 50
    local_poly_order: int = 2
    projection_normalization: str = "row"
    projection_scales: tuple[float, ...] = (1.0,)
    regressor: str = "adaptive_lasso"
    library_space: str = "z"
    target_anchor: str = "left"
    stlsq_threshold: float = 0.12
    threshold_mode: str = "relative"
    pseudo_blocks: int = 5
    adaptive_gamma: float = 1.0
    ridge_floor: float = 1e-10
    gls_weighting: bool = True
    gls_iterations: int = 1
    diffusion_parameterization: str = "chol"
    diffusion_shrinkage: float = 0.05
    bias_correct: bool = True
    noise_correct: bool = False


WG_SINDY_DEFAULTS = WGSINDyDefaults()


def wg_sindy_defaults() -> dict[str, Any]:
    return asdict(WG_SINDY_DEFAULTS)


def fit_wg_sindy(
    states,
    increments=None,
    *,
    dt: float = 0.01,
    library: Library | None = None,
    state_names=("x", "y"),
    seed: int = 0,
    traj_ids=None,
    **overrides: Any,
) -> GeneratorFit2D:
    """Fit the frozen WG-SINDy estimator.

    Overrides are intentionally scalar/configuration-level controls; the
    mechanism set remains the v6 frozen weak-form stack.
    """

    defaults = wg_sindy_defaults()
    defaults.update(overrides)
    extra_regression_kw = defaults.pop("regression_kw", None)
    regression_kw = {
        "stlsq_threshold": defaults.pop("stlsq_threshold"),
        "threshold_mode": defaults.pop("threshold_mode"),
        "pseudo_blocks": defaults.pop("pseudo_blocks"),
        "gamma": defaults.pop("adaptive_gamma"),
        "ridge_floor": defaults.pop("ridge_floor"),
    }
    if extra_regression_kw:
        regression_kw.update(dict(extra_regression_kw))
    library_space = defaults.pop("library_space")
    precomputed_projection = defaults.pop("precomputed_projection", None)
    return_projection = defaults.pop("return_projection", False)
    return fit_generator_2d(
        states,
        increments=increments,
        dt=dt,
        library=library,
        state_names=state_names,
        seed=seed,
        traj_ids=traj_ids,
        regression_kw=regression_kw,
        library_space=library_space,
        precomputed_projection=precomputed_projection,
        return_projection=return_projection,
        **defaults,
    )


__all__ = ["WG_SINDY_DEFAULTS", "WGSINDyDefaults", "fit_wg_sindy", "wg_sindy_defaults"]
