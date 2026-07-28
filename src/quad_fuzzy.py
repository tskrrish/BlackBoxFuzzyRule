"""
quad_fuzzy.py — library for the geometric prototype-rule explainer.

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
# (b) PCA embedding: flatten 4D -> 2D 
# ---------------------------------------------------------------------------
def embed_2d(df, cols=("a", "b", "c", "d")):
    """Project the 4D data to 2D. Returns (pca, xy).
    pca.explained_variance_ratio_.sum() tells you how much you kept."""
    pca = PCA(n_components=2)
    xy = pca.fit_transform(df[list(cols)])
    return pca, xy



# ---------------------------------------------------------------------------
# (c) pick corners
# ---------------------------------------------------------------------------


def pick_corners(xy, k, seed=0):
    """Farthest-point sampling: k well-spread corners from the cloud xy."""
    rng = np.random.default_rng(seed)
    n = xy.shape[0]
    chosen = [int(rng.integers(n))]                      # 1. random start
    dist = np.linalg.norm(xy - xy[chosen[0]], axis=1)    # dist from every point to that corner
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))                       # (A) point farthest from chosen set
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[nxt], axis=1))  # (B) update nearest-corner dist
    return chosen


# ---------------------------------------------------------------------------
# (d) barycentric coordinates
# ---------------------------------------------------------------------------


def barycentric_triangle(p, A, B, C):
    """Weights of p as a blend of triangle corners A, B, C.
    Returns (wA, wB, wC), summing to 1. A negative weight => p is OUTSIDE."""
    A, B, C, p = map(np.asarray, (A, B, C, p))
    # Solve  wA*A + wB*B + wC*C = p  with  wA+wB+wC = 1.
    # Substitute wA = 1 - wB - wC, giving a 2x2 linear system in (wB, wC):
    #   wB*(B - A) + wC*(C - A) = p - A
    M = np.column_stack([B - A, C - A])   # 2x2 matrix
    wB, wC = np.linalg.solve(M, p - A)     # solve the system
    wA = 1 - wB - wC
    return wA, wB, wC


def quad_weights(p, corners, tol=1e-9):
    """
    Weights over 4 corners A,B,C,D, split along diagonal A-C.
    Returns dict {A,B,C,D} summing to 1, or None if p is outside the quad.
    """
    A, B, C, D = corners
    # try the A-B-C half
    wA, wB, wC = barycentric_triangle(p, A, B, C)
    if min(wA, wB, wC) >= -tol:                     # inside ABC?
        return {"A": wA, "B": wB, "C": wC, "D": 0.0}
    # else try the A-C-D half
    wA, wC, wD = barycentric_triangle(p, A, C, D)
    if min(wA, wC, wD) >= -tol:                     # inside ACD?
        return {"A": wA, "B": 0.0, "C": wC, "D": wD}
    return None                                     # outside both -> no honest explanation



def delaunay_weights(p, tri, n_corners):
    """
    Weights of p over ALL corners, via Delaunay triangulation `tri`.
    Returns a length-n_corners array summing to 1, or None if p is outside.
    """
    s = tri.find_simplex(p)
    if s < 0:                                    # outside every triangle
        return None
    # barycentric coords of p within triangle #s
    T = tri.transform[s, :2]                     # 2x2 transform for this simplex
    r = tri.transform[s, 2]                      # reference vertex offset
    bary = T.dot(p - r)                          # first 2 barycentric coords
    bary = np.append(bary, 1 - bary.sum())       # third coord (they sum to 1)
    # scatter these 3 weights onto the full corner list
    w = np.zeros(n_corners)
    verts = tri.simplices[s]                      # the 3 corner indices of triangle s
    w[verts] = bary
    return w