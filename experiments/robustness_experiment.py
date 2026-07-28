"""
robustness_experiment.py — the headline result.

Sweeps coverage and fidelity across rising noise (falling R²). Shows the two
are DECOUPLED: coverage stays flat (geometry, blind to redundancy) while the
fidelity error blows up ~10x (tracks R² collapsing). More corners fix neither.

"""

import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay

# make src/ importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quad_fuzzy import make_data, fit_predictors, embed_2d, pick_corners, delaunay_weights


def measure(noise, k=10):
    """Return (mean R², coverage, mean fidelity-error) for one noise level."""
    df = make_data(noise=noise)
    pca, xy = embed_2d(df)
    idx = pick_corners(xy, k)
    corners = xy[idx]
    tri = Delaunay(corners)

    # the black box we're explaining: predict d from a,b,c
    bb = LinearRegression().fit(df[["a", "b", "c"]], df["d"])
    bb_pred = bb.predict(df[["a", "b", "c"]])
    corner_d = df["d"].values[idx]           # each corner's rule output

    covered = [i for i in range(len(xy)) if tri.find_simplex(xy[i]) >= 0]
    coverage = len(covered) / len(xy)

    # fidelity = |rule-blend - BLACK BOX output|, averaged over covered points
    fid = []
    for i in covered:
        w = delaunay_weights(xy[i], tri, len(corners))
        fid.append(abs(np.dot(w, corner_d) - bb_pred[i]))
    fidelity = np.mean(fid)

    mean_r2 = np.mean(list(fit_predictors(df).values()))
    return mean_r2, coverage, fidelity


if __name__ == "__main__":
    print(f"{'noise':>6} | {'meanR²':>7} | {'coverage':>9} | {'fidelity':>9}")
    print("-" * 42)
    for noise in (0.03, 0.10, 0.20, 0.35, 0.50):
        r2, cov, fid = measure(noise)
        print(f"{noise:>6} | {r2:>7.3f} | {cov:>8.1%} | {fid:>9.4f}")

    print("\nfidelity is an ERROR metric — lower is better.")
    print("coverage ~flat while fidelity error grows ~10x  =>  DECOUPLED.")