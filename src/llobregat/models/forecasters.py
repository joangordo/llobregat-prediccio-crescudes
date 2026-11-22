"""Single-horizon forecasting models: Ridge (linear reference) and LightGBM.

Both take the future flow Q(t+h) as the target and transform it internally with the current
flow Q(t) (by default log(Q(t+h)+ε) − log(Q(t)+ε)): they learn the relative change, which
is more stable between low flows and floods. Forecasts are returned in m³/s. Transformed
forecasts are clipped to the range seen in training, because extrapolating on a log scale
can give absurd values.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from llobregat.features import forward_target, inverse_target

LGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
}


def mse(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.mean((a[ok] - b[ok]) ** 2))


@dataclass
class _Transformed:
    transform: str = "log_ratio"
    eps: float = 0.1
    z_range: tuple[float, float] = field(default=(-np.inf, np.inf), init=False)

    def _z(self, y, q) -> np.ndarray:
        return forward_target(y, q, self.transform, self.eps)

    def _back(self, z, q) -> np.ndarray:
        z = np.clip(np.asarray(z, dtype=float), *self.z_range)
        return inverse_target(z, q, self.transform, self.eps)

    def _set_range(self, z: np.ndarray) -> None:
        self.z_range = (float(np.nanmin(z)), float(np.nanmax(z)))


@dataclass
class RidgeForecaster(_Transformed):
    """Linear regression with L2 penalty. Gaps: training median.

    The penalty is chosen on the inner validation (error in m³/s) and the model is then
    refitted on the whole training period.
    """

    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    alpha: float | None = field(default=None, init=False)
    pipeline: object = field(default=None, init=False, repr=False)

    def _make(self, alpha: float):
        # With highly correlated features and little penalty, scikit-learn warns that the
        # matrix is ill-conditioned: this is expected, and the validation picks a larger
        # penalty if needed
        warnings.filterwarnings("ignore", message="Ill-conditioned matrix")
        warnings.filterwarnings("ignore", message="An ill-conditioned matrix")
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=alpha),
        )

    def tune(self, X_tr, y_tr, q_tr, X_val, y_val, q_val) -> RidgeForecaster:
        z = self._z(y_tr, q_tr)
        self._set_range(z)
        best = None
        for a in self.alphas:
            m = self._make(a).fit(X_tr, z)
            err = mse(y_val, self._back(m.predict(X_val), q_val))
            if best is None or err < best[0]:
                best = (err, a)
        self.alpha = best[1]
        return self

    def fit(self, X, y, q) -> RidgeForecaster:
        z = self._z(y, q)
        self._set_range(z)
        self.pipeline = self._make(self.alpha if self.alpha is not None else 1.0).fit(X, z)
        return self

    def predict(self, X, q) -> np.ndarray:
        return self._back(self.pipeline.predict(X), q)


@dataclass
class LGBMForecaster(_Transformed):
    """LightGBM on the transformed target. Gaps are handled natively."""

    params: dict = field(default_factory=lambda: dict(LGBM_PARAMS))
    max_rounds: int = 3000
    early_stopping: int = 100
    best_iteration: int | None = field(default=None, init=False)
    booster: object = field(default=None, init=False, repr=False)
    features: list[str] = field(default_factory=list, init=False)

    def _dataset(self, X: pd.DataFrame, z, weight=None, reference=None):
        import lightgbm as lgb

        return lgb.Dataset(
            X.to_numpy(dtype=np.float32),
            label=z,
            weight=weight,
            feature_name=[_safe(c) for c in X.columns],
            reference=reference,
            free_raw_data=True,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y,
        q,
        X_val: pd.DataFrame | None = None,
        y_val=None,
        q_val=None,
        weight=None,
        rounds: int | None = None,
    ) -> LGBMForecaster:
        """With validation: stop when the validation stops improving (and keep that
        iteration). Without validation: train exactly `rounds` iterations."""
        import lightgbm as lgb

        self.features = list(X.columns)
        z = self._z(y, q)
        self._set_range(z)
        train = self._dataset(X, z, weight)
        if X_val is not None:
            valid = self._dataset(X_val, self._z(y_val, q_val), reference=train)
            self.booster = lgb.train(
                self.params,
                train,
                num_boost_round=self.max_rounds,
                valid_sets=[valid],
                callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
            )
            self.best_iteration = int(self.booster.best_iteration or self.max_rounds)
        else:
            n = rounds or self.best_iteration or self.max_rounds
            self.booster = lgb.train(self.params, train, num_boost_round=n)
            self.best_iteration = n
        return self

    def predict(self, X: pd.DataFrame, q) -> np.ndarray:
        z = self.booster.predict(
            X[self.features].to_numpy(dtype=np.float32), num_iteration=self.best_iteration
        )
        return self._back(z, q)

    def importance(self) -> pd.DataFrame:
        gain = self.booster.feature_importance("gain", iteration=self.best_iteration)
        split = self.booster.feature_importance("split", iteration=self.best_iteration)
        df = pd.DataFrame({"feature": self.features, "gain": gain, "splits": split})
        df["gain_pct"] = 100 * df["gain"] / max(df["gain"].sum(), 1e-12)
        return df.sort_values("gain", ascending=False).reset_index(drop=True)


def _safe(name: str) -> str:
    """LightGBM does not accept some characters in feature names."""
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
