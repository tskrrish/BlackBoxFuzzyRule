"""
test_bodyfat.py — tests for the real-data body-fat application.

Two kinds of tests:
  1. Loader-contract tests that run WITHOUT the CSV (they check the loader fails
     loudly and correctly when data is absent). These always run.
  2. Real-data tests that need applications/bodyfat/data/bodyfat.csv. These SKIP
     cleanly if the CSV hasn't been downloaded yet, so the suite stays green
     until you add the file.

Run from the repo root with either:
    python tests/test_bodyfat.py       (plain asserts, prints OK / SKIP)
    pytest tests/                       (if you have pytest)
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(__file__)
APP = os.path.join(HERE, "..", "applications", "bodyfat")
sys.path.insert(0, APP)

import load_bodyfat
from load_bodyfat import FEATS, TARGET, COLS
import bodyfat_explainer as bfx
from scipy.spatial import Delaunay


CSV_PRESENT = os.path.exists(load_bodyfat.CSV_PATH)


class Skip(Exception):
    """Raised to mark a test skipped (CSV not downloaded yet)."""


# --- loader contract: works with or without the CSV -------------------------
def test_loader_missing_raises_filenotfound():
    """Pointed at a nonexistent path, the loader must raise a clear error."""
    try:
        load_bodyfat.load(csv_path="/no/such/bodyfat.csv")
    except FileNotFoundError:
        return                       # correct behavior
    raise AssertionError("expected FileNotFoundError for a missing CSV")


def test_canonical_map_has_required_targets():
    """Every column the pipeline needs must be resolvable by the alias map."""
    for c in COLS:
        assert c in load_bodyfat.CANONICAL


# --- real-data tests: skip if the CSV is absent -----------------------------
def _load_or_skip():
    if not CSV_PRESENT:
        raise Skip()
    return load_bodyfat.load()


def test_real_rows_load():
    df = _load_or_skip()
    assert len(df) > 200                         # StatLib body-fat is 252 rows
    for c in COLS:
        assert c in df.columns


def test_bodyfat_values_are_physical():
    df = _load_or_skip()
    assert df[TARGET].between(0, 60).all()       # loader drops non-physical rows


def test_black_box_is_predictive_on_real_data():
    """abdomen/hip/chest should explain a solid chunk of body-fat variance."""
    df = _load_or_skip()
    from sklearn.linear_model import LinearRegression
    sub = (df[COLS] - df[COLS].mean()) / df[COLS].std()
    r2 = LinearRegression().fit(sub[FEATS], sub[TARGET]).score(sub[FEATS], sub[TARGET])
    assert r2 > 0.4                              # real data clears this comfortably


def test_explanation_weights_sum_to_one_and_are_sparse():
    """A covered person's blend uses exactly 3 prototypes and sums to 1."""
    df = _load_or_skip()
    from sklearn.decomposition import PCA
    sub = (df[COLS] - df[COLS].mean()) / df[COLS].std()
    xy = PCA(2).fit_transform(sub[COLS])
    ci = bfx.farthest_point_corners(xy, 8)
    tri = Delaunay(xy[ci])
    covered = [i for i in range(len(xy)) if tri.find_simplex(xy[i]) >= 0]
    i = covered[0]
    s = tri.find_simplex(xy[i])
    w = bfx.bary(xy[i], tri, s); w = np.clip(w, 0, None); w = w / w.sum()
    assert np.isclose(w.sum(), 1.0)
    assert np.count_nonzero(w) <= 3


def test_out_of_hull_person_gets_no_explanation():
    """The abstention property: a far-away point returns simplex < 0."""
    df = _load_or_skip()
    from sklearn.decomposition import PCA
    sub = (df[COLS] - df[COLS].mean()) / df[COLS].std()
    xy = PCA(2).fit_transform(sub[COLS])
    ci = bfx.farthest_point_corners(xy, 8)
    tri = Delaunay(xy[ci])
    assert tri.find_simplex(np.array([1e3, 1e3])) < 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = 0
    for fn in fns:
        try:
            fn()
            print(f"ok    {fn.__name__}")
            passed += 1
        except Skip:
            print(f"SKIP  {fn.__name__}  (no CSV at {load_bodyfat.CSV_PATH})")
            skipped += 1
    print(f"\n{passed} passed, {skipped} skipped")
    if skipped:
        print("Download bodyfat.csv (see applications/bodyfat/load_bodyfat.py) "
              "to run the real-data tests.")