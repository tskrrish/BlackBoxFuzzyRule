"""
test_quad_fuzzy.py — property tests for the explainer library.

Run from the repo root with either:
    python tests/test_quad_fuzzy.py       (plain asserts, prints OK)
    pytest tests/                          (if you have pytest)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quad_fuzzy import (
    make_data, fit_predictors, embed_2d, pick_corners,
    barycentric_triangle, delaunay_weights,
)
from scipy.spatial import Delaunay


# --- barycentric_triangle: the three cases you verified by hand ---
def test_barycentric_on_vertex():
    A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    w = barycentric_triangle(A, A, B, C)          # p sits exactly on corner A
    assert np.allclose(w, (1, 0, 0))              # a vertex -> all weight on that corner


def test_barycentric_centroid():
    A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    centroid = (A + B + C) / 3
    w = barycentric_triangle(centroid, A, B, C)
    assert np.allclose(w, (1/3, 1/3, 1/3))        # dead center -> equal blend


def test_barycentric_outside_is_negative():
    A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    w = barycentric_triangle([1, 1], A, B, C)     # a point outside the triangle
    assert min(w) < 0                             # outside -> at least one negative weight


def test_barycentric_sums_to_one():
    A, B, C = np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    for p in ([0.2, 0.3], [1, 1], [-0.5, 0.4]):
        assert np.isclose(sum(barycentric_triangle(p, A, B, C)), 1.0)


# --- delaunay_weights: sparsity + outside-returns-None ---
def test_delaunay_sparsity_and_sum():
    df = make_data()
    _, xy = embed_2d(df)
    idx = pick_corners(xy, 10)
    corners = xy[idx]
    tri = Delaunay(corners)
    covered = [i for i in range(len(xy)) if tri.find_simplex(xy[i]) >= 0]
    w = delaunay_weights(xy[covered[0]], tri, len(corners))
    assert np.count_nonzero(w) == 3               # a point lands in ONE triangle -> 3 corners fire
    assert np.isclose(w.sum(), 1.0)


def test_delaunay_outside_returns_none():
    df = make_data()
    _, xy = embed_2d(df)
    corners = xy[pick_corners(xy, 10)]
    tri = Delaunay(corners)
    assert delaunay_weights(np.array([100.0, 100.0]), tri, 10) is None


# --- fit_predictors: high R² on clean data, low on noisy ---
def test_fit_predictors_regime():
    clean = np.mean(list(fit_predictors(make_data(noise=0.03)).values()))
    noisy = np.mean(list(fit_predictors(make_data(noise=0.5)).values()))
    assert clean > 0.8      # thin sheet
    assert noisy < 0.3      # fat blob


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")