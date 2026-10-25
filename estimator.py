"""Scikit-learn models that estimate cost, time and success probability.

These models are *predictors* (the "Cost & Time" and "Risk" analysts use their
output).  They do not choose the delivery option - the RL agent does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import TRAFFIC_CODE


def _cost_x(d, w, tr, wr):
    return [d, w, tr, wr, d * tr]


def _time_x(d, w, tr, wr, no_driver):
    return [d, w, tr, wr, d * tr, d * wr, no_driver]


class OptionEstimator:
    """Learns, per delivery option, how cost / time / success depend on the situation."""

    def __init__(self) -> None:
        self.options: list[str] = []
        self.cost_models: dict[str, LinearRegression] = {}
        self.time_models: dict[str, LinearRegression] = {}
        self.success_model = None
        self.success_constant: float | None = None
        self.max_weight: dict[str, float] = {}
        self.metrics: dict[str, float] = {}

    # ---------------------------------------------------------------- fit
    @staticmethod
    def _matrices(df: pd.DataFrame):
        d = df["distance_km"].to_numpy(float)
        w = df["package_weight"].to_numpy(float)
        tr = df["traffic_level"].map(TRAFFIC_CODE).to_numpy(float)
        wr = df["weather_risk"].to_numpy(float)
        no_driver = 1.0 - df["driver_available"].to_numpy(float)
        return d, w, tr, wr, no_driver

    def _success_matrix(self, d, w, tr, wr, rating, drv, option_idx):
        k = len(self.options)
        base = np.column_stack([d / 10.0, w / 10.0, tr, wr, rating, drv])
        onehot = np.zeros((len(d), k))
        onehot[np.arange(len(d)), option_idx] = 1.0
        return np.hstack([base, onehot, onehot * wr[:, None], onehot * tr[:, None]])

    def _fit_models(self, df: pd.DataFrame) -> None:
        d, w, tr, wr, nd = self._matrices(df)
        opt = df["delivery_option"].to_numpy()
        for option in self.options:
            m = opt == option
            xc = np.column_stack(_cost_x(d[m], w[m], tr[m], wr[m]))
            xt = np.column_stack(_time_x(d[m], w[m], tr[m], wr[m], nd[m]))
            self.cost_models[option] = LinearRegression().fit(xc, df.loc[m, "delivery_cost"])
            self.time_models[option] = LinearRegression().fit(xt, df.loc[m, "estimated_time"])
            self.max_weight[option] = float(df.loc[m, "package_weight"].max())

        y = df["delivery_success"].to_numpy(int)
        if len(np.unique(y)) < 2:
            self.success_constant = float(y.mean())
            self.success_model = None
            return
        self.success_constant = None
        idx = np.array([self.options.index(o) for o in opt])
        x = self._success_matrix(d, w, tr, wr, df["driver_rating"].to_numpy(float),
                                 df["driver_available"].to_numpy(float), idx)
        self.success_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self.success_model.fit(x, y)

    def fit(self, df: pd.DataFrame) -> "OptionEstimator":
        self.options = sorted(df["delivery_option"].unique().tolist())
        # 1) hold-out run just to report honest quality numbers
        try:
            train, test = train_test_split(df, test_size=0.2, random_state=0)
            self._fit_models(train)
            self.metrics = self._evaluate(test)
        except Exception:  # metrics are a nice-to-have, never block training
            self.metrics = {}
        # 2) final models use all rows
        self._fit_models(df)
        return self

    def _evaluate(self, test: pd.DataFrame) -> dict[str, float]:
        d, w, tr, wr, nd = self._matrices(test)
        opt = test["delivery_option"].to_numpy()
        cost_pred = np.zeros(len(test))
        time_pred = np.zeros(len(test))
        for option in self.options:
            m = opt == option
            if not m.any():
                continue
            cost_pred[m] = self.cost_models[option].predict(np.column_stack(_cost_x(d[m], w[m], tr[m], wr[m])))
            time_pred[m] = self.time_models[option].predict(np.column_stack(_time_x(d[m], w[m], tr[m], wr[m], nd[m])))
        out = {
            "cost_r2": float(r2_score(test["delivery_cost"], cost_pred)),
            "time_r2": float(r2_score(test["estimated_time"], time_pred)),
        }
        if self.success_model is not None:
            idx = np.array([self.options.index(o) for o in opt])
            x = self._success_matrix(d, w, tr, wr, test["driver_rating"].to_numpy(float),
                                     test["driver_available"].to_numpy(float), idx)
            out["success_accuracy"] = float(accuracy_score(test["delivery_success"], self.success_model.predict(x)))
        return out

    # ----------------------------------------------------------- predict
    def is_feasible_weight(self, option: str, weight: float) -> bool:
        return weight <= self.max_weight.get(option, 0.0) * 1.25

    def estimate(self, ctx: dict, option: str) -> dict[str, float]:
        """Predict cost, time and success probability for one option and one situation."""
        d = float(ctx["distance_km"])
        w = float(ctx["package_weight"])
        tr = float(TRAFFIC_CODE[ctx["traffic_level"]])
        wr = float(ctx["weather_risk"])
        drv = 1.0 if ctx.get("driver_available", True) else 0.0
        rating = float(ctx.get("driver_rating", 4.3))

        cost = float(self.cost_models[option].predict(np.array([_cost_x(d, w, tr, wr)]))[0])
        time_min = float(self.time_models[option].predict(np.array([_time_x(d, w, tr, wr, 1.0 - drv)]))[0])
        if self.success_model is None:
            p = float(self.success_constant if self.success_constant is not None else 0.9)
        else:
            idx = np.array([self.options.index(option)])
            x = self._success_matrix(np.array([d]), np.array([w]), np.array([tr]), np.array([wr]),
                                     np.array([rating]), np.array([drv]), idx)
            p = float(self.success_model.predict_proba(x)[0, 1])
        return {
            "cost": max(0.5, cost),
            "time_min": max(5.0, time_min),
            "p_success": float(np.clip(p, 0.05, 0.995)),
        }
