# Geometric Prototype-Rule Explanation of a Black-Box Regressor

Explaining a black-box regressor as a barycentric blend of geometric "prototype"
corners — a Delaunay/simplex-based Takagi–Sugeno fuzzy-rule construction — and
studying when that explanation is trustworthy.

## One-line summary

Every prediction is reframed as "this point is X% prototype A + Y% prototype B + …",
where the weights are barycentric coordinates (they sum to 1) over a simplex of
prototype corners. Each corner has a human-readable label, so the mixture of
activated rules becomes the explanation.

## The finding

Coverage and fidelity are decoupled. Coverage (can a point be explained at all?)
is a robust geometric property — ~8–10 corners cover 90%+ of points regardless of
data structure. Fidelity (is the explanation faithful to the model?) collapses ~10×
once the columns stop being mutually predictable, and adding corners does not fix it.
So high coverage can give a false sense of security. The load-bearing assumption is
that the data is genuinely low-dimensional / redundant (high R²).

## What is / isn't novel

- Established: Delaunay-triangulation-based Takagi–Sugeno fuzzy models using
  barycentric coordinates as rule firing strengths; prototype-based explanation.
- Novel : the coverage/fidelity decoupling on a controlled setup, and
  the observation that coverage alone can mislead.

## Structure

    src/           quad_fuzzy.py, tetra_fuzzy.py
    experiments/   sweep_experiment.py, robustness_experiment.py
    notes/         journey.md
    tests/         (to add)
    explore.ipynb  guided walkthrough

## Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
