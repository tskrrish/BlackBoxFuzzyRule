"""
decoupling_multidataset.py — coverage/fidelity DECOUPLING on REAL data.

This is the real-data version of experiments/robustness_experiment.py's headline
result. Instead of sweeping synthetic noise on one dataset, it spans the
redundancy axis using THREE genuinely different real datasets:

    bodyfat   (HIGH mutual predictability)  — local CSV via load_bodyfat
    energy    (MEDIUM)                       — UCI id=242 via ucimlrepo
    concrete  (LOW)                          — UCI id=165 via ucimlrepo

The claim being tested (findings §2.1): coverage and fidelity are DECOUPLED.
  - coverage  = fraction of points inside some simplex (can it be explained?)
                a GEOMETRIC property; should stay high regardless of redundancy.
  - fidelity  = mean |rule-blend - black-box output| (is the explanation faithful?)
                depends on redundancy; should degrade as mutual-R^2 falls.

If coverage stays ~flat while fidelity error rises as mutual-R^2 drops across the
three datasets, decoupling holds on real data.

--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------
    pip install ucimlrepo        # fetches concrete + energy straight into pandas
The bodyfat CSV must already be at applications/bodyfat/data/bodyfat.csv.

Run from the repo root:
    python experiments/decoupling_multidataset.py

Add --seeds N for multi-seed error bars (findings §5 asks for this):
    python experiments/decoupling_multidataset.py --seeds 5
--------------------------------------------------------------------------------
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from scipy.spatial import Delaunay

# reach the bodyfat loader in applications/
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "applications", "bodyfat"))


# ---------------------------------------------------------------------------
# REAL dataset loaders — each returns (df, feats, target, label)
# The three predictor columns are chosen to land at different mutual-R^2 levels;
# the script MEASURES and reports the actual mutual-R^2 so the spread is verified,
# not assumed (findings §4.1: redundancy is a property of the chosen column set).
# ---------------------------------------------------------------------------
def load_body():
    import load_bodyfat
    df = load_bodyfat.load()
    return df, ["abdomen", "hip", "chest"], "bodyfat", "BODY (expect high redundancy)"


def load_energy():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=242)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    # columns are X1..X8 / Y1,Y2 (or descriptive names depending on version)
    df = pd.concat([X, y], axis=1)
    df.columns = [str(c).strip() for c in df.columns]
    # normalize to the names we use; X2=surface, X3=wall, X4=roof, Y1=heating load
    rename = {"X2": "surface", "X3": "wall", "X4": "roof", "Y1": "heat",
              "Surface_Area": "surface", "Wall_Area": "wall", "Roof_Area": "roof",
              "Heating_Load": "heat", "Y1 ": "heat"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df, ["surface", "wall", "roof"], "heat", "ENERGY (expect medium redundancy)"


def load_concrete():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=165)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    df.columns = [str(c).strip() for c in df.columns]
    # pick near-independent recipe knobs -> low mutual predictability
    rename = {"Cement": "cement", "Water": "water", "Fine Aggregate": "fine",
              "Fine_Aggregate": "fine", "Concrete compressive strength": "strength",
              "Strength": "strength"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df, ["cement", "water", "fine"], "strength", "CONCRETE (expect low redundancy)"


# ---------------------------------------------------------------------------
# metric machinery — identical to gap12_multidataset.py
# ---------------------------------------------------------------------------
def fpc(points, k, seed=0):
    rng = np.random.default_rng(seed)
    nn = points.shape[0]
    ch = [int(rng.integers(nn))]
    d = np.linalg.norm(points - points[ch[0]], axis=1)
    for _ in range(k - 1):
        nx = int(np.argmax(d)); ch.append(nx)
        d = np.minimum(d, np.linalg.norm(points - points[nx], axis=1))
    return ch


def bary(p, tri, s):
    T = tri.transform[s, :2]; r = tri.transform[s, 2]
    b = T.dot(p - r); return np.append(b, 1 - b.sum())


def run(k, xy, bbp, cd, seed=0):
    ci = fpc(xy, k, seed); corners = xy[ci]; cvals = cd[ci]; tri = Delaunay(corners)
    inside = 0; fe = []; dom = []
    for i in range(len(xy)):
        s = tri.find_simplex(xy[i])
        if s < 0: continue
        inside += 1
        w = bary(xy[i], tri, s); w = np.clip(w, 0, None); w = w / w.sum()
        v = tri.simplices[s]
        fe.append(abs(np.dot(w, cvals[v]) - bbp[i])); dom.append(w.max())
    return inside / len(xy), (np.mean(fe) if fe else np.nan), (np.mean(dom) if dom else np.nan)


def loo_r2(sub, cols):
    """Mutual predictability of the PREDICTOR SET: each predictor from the others."""
    s = []
    for c in cols:
        y = sub[c].values
        X = sub[[x for x in cols if x != c]].values
        s.append(cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean())
    return float(np.mean(s))


def analyze(df, feats, target, label, seed=0):
    use = feats + [target]
    sub = df[use].dropna().copy()
    sub = (sub - sub.mean()) / sub.std()
    setR2 = loo_r2(sub, feats)                      # <-- the redundancy axis
    rf = RandomForestRegressor(n_estimators=200, random_state=seed)
    rf.fit(sub[feats], sub[target])
    bbp = rf.predict(sub[feats])
    rf_r2 = cross_val_score(rf, sub[feats], sub[target], cv=5, scoring="r2").mean()
    pca = PCA(2); xy = pca.fit_transform(sub[use]); cd = sub[target].values
    rows = {k: run(k, xy, bbp, cd, seed) for k in [4, 6, 8, 10, 15, 20, 30, 40]}
    return {"label": label, "setR2": setR2, "rf_r2": rf_r2,
            "var2d": pca.explained_variance_ratio_.sum(), "rows": rows}


def analyze_multiseed(loader, seeds):
    df, feats, target, label = loader()
    runs = [analyze(df, feats, target, label, seed=s) for s in range(seeds)]
    # aggregate: mean +/- std over seeds for each K
    agg = {"label": label,
           "setR2": np.mean([r["setR2"] for r in runs]),
           "rf_r2": np.mean([r["rf_r2"] for r in runs]),
           "var2d": np.mean([r["var2d"] for r in runs]), "rows": {}}
    for k in [4, 6, 8, 10, 15, 20, 30, 40]:
        cov = [r["rows"][k][0] for r in runs]
        fid = [r["rows"][k][1] for r in runs]
        sp = [r["rows"][k][2] for r in runs]
        agg["rows"][k] = (np.mean(cov), np.std(cov), np.mean(fid), np.std(fid),
                          np.mean(sp), np.std(sp))
    return agg, len(runs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of random seeds (>=2 gives error bars)")
    args = ap.parse_args()

    loaders = [("body", load_body), ("energy", load_energy), ("concrete", load_concrete)]
    results = []
    for name, loader in loaders:
        try:
            if args.seeds > 1:
                r, nseeds = analyze_multiseed(loader, args.seeds)
            else:
                df, feats, target, label = loader()
                r = analyze(df, feats, target, label)
                nseeds = 1
            results.append(r)
        except Exception as e:
            print(f"\n[skipped {name}] {type(e).__name__}: {e}")
            print("  (concrete/energy need `pip install ucimlrepo`; "
                  "bodyfat needs the CSV in place.)")
            continue

        print(f"\n=== {r['label']} ===")
        print(f"predictor mutual R^2 (redundancy) : {r['setR2']:.3f}")
        print(f"black-box R^2 (RF)                : {r['rf_r2']:.3f}")
        print(f"2D variance kept                  : {r['var2d']:.3f}")
        if args.seeds > 1:
            print(f"{'#c':>4} | {'coverage':>14} | {'fidelity':>16} | {'sparsity':>12}")
            for k in [4, 8, 10, 20, 40]:
                cM, cS, fM, fS, sM, sS = r["rows"][k]
                print(f"{k:>4} | {cM:>6.1%} +-{cS:>4.1%} | {fM:>7.4f} +-{fS:>6.4f} | {sM:>5.2f} +-{sS:>4.2f}")
        else:
            print(f"{'#c':>4} | {'cov':>6} | {'fid':>7} | {'spars':>6}")
            for k in [4, 8, 10, 20, 40]:
                cov, fid, sp = r["rows"][k]
                print(f"{k:>4} | {cov:>5.1%} | {fid:>7.4f} | {sp:>5.2f}")

    if len(results) >= 2:
        print("\n\n========= DECOUPLING ACROSS REAL DATASETS =========")
        seedtag = f" (mean of {args.seeds} seeds)" if args.seeds > 1 else ""
        print(f"{'dataset':<34} {'mutR2':>6} {'rfR2':>6} {'cov@10':>7} {'fid@10':>7} {'fid@40':>7}{seedtag}")
        for r in results:
            if args.seeds > 1:
                c10 = r["rows"][10]; c40 = r["rows"][40]
                print(f"{r['label']:<34} {r['setR2']:>6.3f} {r['rf_r2']:>6.3f} "
                      f"{c10[0]:>6.1%} {c10[2]:>7.4f} {c40[2]:>7.4f}")
            else:
                c10 = r["rows"][10]; c40 = r["rows"][40]
                print(f"{r['label']:<34} {r['setR2']:>6.3f} {r['rf_r2']:>6.3f} "
                      f"{c10[0]:>6.1%} {c10[1]:>7.4f} {c40[1]:>7.4f}")
        print("\nDECOUPLING holds if: coverage@10 stays ~flat/high across rows")
        print("while fidelity error rises as mutual-R^2 falls (body -> energy -> concrete).")