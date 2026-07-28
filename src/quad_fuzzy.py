"""
quad_fuzzy.py — library for the geometric prototype-rule explainer.

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


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
