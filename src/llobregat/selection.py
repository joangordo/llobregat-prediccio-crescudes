"""Source selection within each fold: grouped permutation importance.

A group is a source of information: a gauging station (all its lags), a rain gauge (all its
sums), a reservoir, a sub-basin... For each group, its rows in the inner validation are
shuffled at random (all the columns of the group with the same permutation, so as not to
break their internal relation) and the increase of the error in m³/s is measured. A source
whose information can be destroyed without making anything worse adds nothing to the model
and is dropped.

It uses the inner validation (training years) of each fold: the test plays no part.
Known limitation: if two sources carry the same information, shuffling only one of them
underestimates its importance. That is why the criterion is conservative (only sources that
make nothing worse are dropped) and it can be switched off for comparison.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from llobregat.analysis import ALWAYS_KEEP_GROUPS, feature_group


def group_permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y,
    groups: dict[str, list[str]],
    n_repeats: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Relative increase of the MSE (in m³/s) when each group is shuffled."""
    y = np.asarray(y, dtype=float)
    base = max(float(np.nanmean((predict(X) - y) ** 2)), 1e-12)
    rng = np.random.default_rng(seed)
    rows = []
    for g, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[perm]
            deltas.append(float(np.nanmean((predict(Xp) - y) ** 2)) / base - 1)
        rows.append(
            {
                "group": g,
                "n_features": len(cols),
                "mse_increase": float(np.mean(deltas)),
                "mse_increase_min": float(np.min(deltas)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("mse_increase", ascending=False).reset_index(drop=True)


def select_groups(
    importance: pd.DataFrame, tol: float = 0.0, always_keep=ALWAYS_KEEP_GROUPS
) -> list[str]:
    """Groups that are kept: those that increase the MSE by more than tol (in every
    repetition) and those that are always kept."""
    keep = importance.loc[importance["mse_increase_min"] > tol, "group"].tolist()
    keep += [g for g in importance["group"] if g in always_keep and g not in keep]
    return keep


def columns_of(groups_kept: list[str], cols: list[str]) -> list[str]:
    kept = set(groups_kept)
    return [c for c in cols if feature_group(c) in kept]
