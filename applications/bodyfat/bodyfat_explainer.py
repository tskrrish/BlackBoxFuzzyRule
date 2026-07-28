"""
bodyfat_explainer.py — the practical application, now on REAL data.

Explains a body-fat estimate as a blend of body-type prototype-rules, using the
same human-readable vocabulary as the reconstruction study.

Black box:   linear regression   abdomen, hip, chest -> body fat %
Explanation: barycentric weights over a Delaunay triangulation of body-type
             prototypes chosen by farthest-point sampling. Each prototype is a
             REAL person; its label comes from which measurements are high/low.

Run from the repo root:
    python applications/bodyfat/bodyfat_explainer.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)  # so `import load_bodyfat` works from repo root
import load_bodyfat
from load_bodyfat import FEATS, TARGET, COLS


# ---- human-readable vocabulary (unchanged from the reconstruction study) ----
def describe_corner(df, idx, cols, low_q=0.3, high_q=0.7):
    parts = []
    row = df.iloc[idx]
    for col in cols:
        pct = (df[col] < row[col]).mean()
        if pct >= high_q:
            parts.append(f"high {col}")
        elif pct <= low_q:
            parts.append(f"low {col}")
    return ", ".join(parts) if parts else "an average build"


def hedge(wt):
    if wt >= 0.75: return "very strongly"
    if wt >= 0.50: return "strongly"
    if wt >= 0.25: return "moderately"
    if wt >= 0.10: return "slightly"
    return None


def farthest_point_corners(points, k, seed=0):
    rng = np.random.default_rng(seed)
    nn = points.shape[0]
    chosen = [int(rng.integers(nn))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist)); chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return chosen


def bary(p, tri, s):
    T = tri.transform[s, :2]; r = tri.transform[s, 2]
    b = T.dot(p - r); return np.append(b, 1 - b.sum())


def explain(i, df, xy, tri, corner_idx, cvals, feat_names):
    """One person's body-fat estimate, explained as a prototype blend."""
    s = tri.find_simplex(xy[i])
    lines = [f"Person {i} (measurements: " +
             ", ".join(f"{c}={df[c].values[i]:.0f}" for c in FEATS) + "):"]
    if s < 0:
        lines.append("  unlike any prototype on record - no honest explanation available")
        return "\n".join(lines), None
    w = bary(xy[i], tri, s); w = np.clip(w, 0, None); w = w / w.sum()
    verts = tri.simplices[s]
    order = sorted(range(len(verts)), key=lambda j: -w[j])
    for j in order:
        h = hedge(w[j])
        if h:
            gi = corner_idx[verts[j]]
            lines.append(f"  {h} resembles the '{describe_corner(df, gi, FEATS)}' build "
                         f"(bodyfat~{df[TARGET].values[gi]:.0f}%) - {w[j]:.0%}")
    pred = float(np.dot(w, cvals[verts]))
    lines.append(f"  => estimated body fat = {pred:.1f}%  (measured: {df[TARGET].values[i]:.1f}%)")
    return "\n".join(lines), pred


def main():
    df = load_bodyfat.load()   # REAL rows now
    print(f"Loaded {len(df)} real people from StatLib body-fat dataset.\n")

    sub = (df[COLS] - df[COLS].mean()) / df[COLS].std()

    bb = LinearRegression().fit(sub[FEATS], sub[TARGET])
    print(f"Black-box body-fat model R^2: {bb.score(sub[FEATS], sub[TARGET]):.3f}")

    pca = PCA(2); xy = pca.fit_transform(sub[COLS])
    print(f"2D embedding keeps {pca.explained_variance_ratio_.sum():.1%} of variance\n")

    K = 8  # the elbow region
    corner_idx = farthest_point_corners(xy, K)
    tri = Delaunay(xy[corner_idx])
    cvals = df[TARGET].values[corner_idx]

    print(f"{K} body-type prototypes discovered (from REAL people):")
    for j, gi in enumerate(corner_idx):
        print(f"  P{j}: '{describe_corner(df, gi, FEATS)}'  bodyfat~{df[TARGET].values[gi]:.0f}%")

    print("\nExample explanations:")
    for i in (3, 40, 120, 200):
        if i < len(df):
            txt, _ = explain(i, df, xy, tri, corner_idx, cvals, FEATS)
            print(txt); print()

    # confident (single-prototype) vs uncertain (blended)
    conf = []
    for i in range(len(df)):
        s = tri.find_simplex(xy[i])
        if s < 0: continue
        w = bary(xy[i], tri, s); w = np.clip(w, 0, None); w = w / w.sum()
        conf.append((w.max(), i))
    conf.sort()
    if conf:
        print("Most BLENDED (uncertain) explanation:")
        print(explain(conf[0][1], df, xy, tri, corner_idx, cvals, FEATS)[0])
        print("\nMost SINGLE-PROTOTYPE (confident) explanation:")
        print(explain(conf[-1][1], df, xy, tri, corner_idx, cvals, FEATS)[0])


if __name__ == "__main__":
    main()