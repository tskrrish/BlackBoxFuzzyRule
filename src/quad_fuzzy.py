"""
quad_fuzzy.py — library for the geometric prototype-rule explainer.

Explains a black-box regressor as a barycentric blend of geometric prototype
corners over a Delaunay triangulation. Built section by section:
  (a) make_data / fit_predictors  — mutual predictability (R²)
  (b) embed_2d                    — PCA flatten 4D -> 2D
  (c) pick_corners                — farthest-point sampling
  (d) barycentric_triangle        — weights in one triangle (negative = outside)
      quad_weights                — 4 corners via diagonal split
      delaunay_weights            — many corners via Delaunay multi-simplex
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# (a) data + mutual predictability
# ---------------------------------------------------------------------------
def make_data(n=500, noise=0.03, seed=0):
    """Four columns where d is (nearly) determined by a, b, c.
    Raise `noise` to weaken mutual predictability (lowers R²)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, n)
    b = rng.uniform(0, 1, n)
    c = rng.uniform(0, 1, n)
    d = 1 - (a + b + c) / 3 + rng.normal(0, noise, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


def fit_predictors(df):
    """R² for predicting each column from the other three.
    High R² => columns mutually predictable => data is a thin low-dim sheet."""
    cols = df.columns.tolist()
    scores = {}
    for target in cols:
        inputs = [c for c in cols if c != target]
        model = LinearRegression().fit(df[inputs], df[target])
        scores[target] = model.score(df[inputs], df[target])
    return scores


# ---------------------------------------------------------------------------
# (b) PCA embedding: flatten 4D -> 2D (the plane we draw triangles on)
# ---------------------------------------------------------------------------
def embed_2d(df, cols=("a", "b", "c", "d")):
    """Project the 4D data to 2D. Returns (pca, xy).
    pca.explained_variance_ratio_.sum() tells you how much you kept."""
    pca = PCA(n_components=2)
    xy = pca.fit_transform(df[list(cols)])
    return pca, xy


# ---------------------------------------------------------------------------
# (c) pick corner archetypes via farthest-point sampling
# ---------------------------------------------------------------------------
def pick_corners(xy, k, seed=0):
    """Farthest-point sampling: k well-spread corners from the cloud xy.
    `dist` always holds each point's distance to its nearest chosen corner;
    we greedily add whichever point is currently farthest from all corners."""
    rng = np.random.default_rng(seed)
    n = xy.shape[0]
    chosen = [int(rng.integers(n))]
    dist = np.linalg.norm(xy - xy[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[nxt], axis=1))
    return chosen


# ---------------------------------------------------------------------------
# (d) barycentric weights
# ---------------------------------------------------------------------------
def barycentric_triangle(p, A, B, C):
    """Weights of p as a blend of triangle corners A, B, C.
    Returns (wA, wB, wC), summing to 1. A negative weight => p is OUTSIDE."""
    A, B, C, p = map(np.asarray, (A, B, C, p))
    M = np.column_stack([B - A, C - A])
    wB, wC = np.linalg.solve(M, p - A)
    wA = 1 - wB - wC
    return wA, wB, wC


def quad_weights(p, corners, tol=1e-9):
    """Weights over 4 corners A,B,C,D, split along diagonal A-C.
    Returns dict {A,B,C,D} summing to 1, or None if p is outside the quad."""
    A, B, C, D = corners
    wA, wB, wC = barycentric_triangle(p, A, B, C)
    if min(wA, wB, wC) >= -tol:
        return {"A": wA, "B": wB, "C": wC, "D": 0.0}
    wA, wC, wD = barycentric_triangle(p, A, C, D)
    if min(wA, wC, wD) >= -tol:
        return {"A": wA, "B": 0.0, "C": wC, "D": wD}
    return None


def delaunay_weights(p, tri, n_corners):
    """Weights of p over ALL corners via Delaunay triangulation `tri`.
    Returns a length-n_corners array summing to 1 (only 3 nonzero — the
    corners of p's triangle), or None if p is outside every simplex."""
    s = tri.find_simplex(p)
    if s < 0:
        return None
    T = tri.transform[s, :2]
    r = tri.transform[s, 2]
    bary = T.dot(p - r)
    bary = np.append(bary, 1 - bary.sum())
    w = np.zeros(n_corners)
    w[tri.simplices[s]] = bary
    return w


# ---------------------------------------------------------------------------
# self-test:
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for noise in (0.03, 0.5):
        df = make_data(noise=noise)
        r2 = {k: round(v, 3) for k, v in fit_predictors(df).items()}
        print(f"noise={noise}: R²={r2}")

    df = make_data()
    pca, xy = embed_2d(df)
    print(f"\n2D keeps {pca.explained_variance_ratio_.sum():.1%} of variance")

    idx = pick_corners(xy, 10)
    corners = xy[idx]
    tri = Delaunay(corners)
    covered = [i for i in range(len(xy)) if tri.find_simplex(xy[i]) >= 0]
    print(f"coverage @10 corners: {len(covered) / len(xy):.1%}")

    w = delaunay_weights(xy[covered[0]], tri, len(corners))
    print(f"example weights (nonzero={np.count_nonzero(w)}):", np.round(w, 3))