#!/usr/bin/env bash
# Reproduces all results and figures from the paper.
#
# Prerequisites:
#   Python 3.11  —  pip install -e .
#   (optional) latexmk for PDF compilation
#
# Runtime: ~5–10 minutes on a modern laptop.
set -euo pipefail

export PAPER_DIR="paper"

echo "=== Step 1: coefficient recovery (R=32 pooled rerun) ==="
if python3 - <<'PY'
import pandas as pd
from pathlib import Path
expected = {
    "correlated_ou", "coupled_ou", "rotational_ou", "spiral_sink_corr",
    "van_der_pol", "stuart_landau", "brusselator", "duffing",
    "maier_stein", "gradient_potential", "diag_multiplicative",
    "nondiag_cholesky",
}
path = Path("results/coefficient_recovery/coeff_recovery_R32.csv")
raise SystemExit(0 if path.exists() and set(pd.read_csv(path)["system"]) == expected else 1)
PY
then
  echo "  Already complete — skipping."
else
  python3 experiments/coefficient_recovery.py --R 32 --seeds 4 --steps 8000
fi

echo "=== Step 2: build paper package (figures + tables + datasheets) ==="
python3 experiments/build_paper.py --full

echo "=== Step 3: run test suite ==="
python3 -m pytest

echo "=== Step 4: compile PDF (optional) ==="
if command -v latexmk >/dev/null 2>&1; then
  (cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
  echo "  PDF written to paper/main.pdf"
else
  echo "  latexmk not found — upload paper/ to Overleaf to compile."
fi

echo ""
echo "Done. Generated outputs:"
echo "  figures/datasheets/    per-system field plots"
echo "  figures/paper/         method comparison heatmap"
echo "  paper/figures/         all figures referenced by LaTeX"
echo "  results/coefficient_recovery/  coefficient recovery table"
echo "  results/paper/         method comparison CSV"
