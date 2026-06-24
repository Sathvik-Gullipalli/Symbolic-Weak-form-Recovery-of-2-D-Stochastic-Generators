from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src-layout package (src/sde2d) is importable when running pytest
# without an editable install. Computed relative to this file so it is robust
# to the working directory pytest is invoked from.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def quick_fit_kwargs():
    return {
        "n_centers": 25,
        "center_scheme": "quantile_grid",
        "grid_shape": (5, 5),
        "bandwidth_multiplier": 1.5,
        "regressor": "stlsq",
        "regression_kw": {"threshold": 0.02},
        "seed": 7,
    }
