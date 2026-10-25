"""Shared constants, the reward function, CSV validation and small helpers.

Everything that more than one module needs lives here so that the data
generator, the estimator, the RL agent and the UI all use exactly the same
business rules (especially the reward function).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Domain constants
# --------------------------------------------------------------------------
PRIORITIES = ["Low", "Standard", "High", "Urgent"]
CUSTOMER_TYPES = ["Regular", "Premium", "Business"]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]
PACKAGE_TYPES = ["Documents", "Parcel", "Fragile", "Perishable", "Heavy"]

# Weather condition (what the user picks) -> numeric weather risk (0..1)
WEATHER_CONDITIONS = {"Clear": 0.05, "Cloudy": 0.15, "Rain": 0.40, "Storm": 0.80}

TRAFFIC_FACTOR = {"Low": 0.1, "Medium": 0.5, "High": 1.0}
TRAFFIC_CODE = {"Low": 0, "Medium": 1, "High": 2}

# Time (minutes) the customer expects the parcel within, per priority
PROMISED_MINUTES = {"Low": 720, "Standard": 360, "High": 180, "Urgent": 90}
# Penalty (currency units) per minute of lateness, per priority
DELAY_RATE = {"Low": 0.02, "Standard": 0.05, "High": 0.15, "Urgent": 0.35}
# Price multiplier the customer pays for the priority level / customer type
PRIORITY_REVENUE_MULT = {"Low": 0.9, "Standard": 1.0, "High": 1.25, "Urgent": 1.6}
CUSTOMER_REVENUE_MULT = {"Regular": 1.0, "Premium": 1.15, "Business": 1.1}
# How much a happy customer is worth
SATISFACTION_WEIGHT = {"Regular": 20.0, "Premium": 25.0, "Business": 22.0}
# Perishable / fragile goods make lateness more expensive
PACKAGE_DELAY_MULT = {
    "Documents": 1.0,
    "Parcel": 1.0,
    "Fragile": 1.2,
    "Perishable": 1.5,
    "Heavy": 1.0,
}

FAILURE_PENALTY = 25.0   # risk penalty when a delivery fails
COST_PENALTY_RATE = 0.10  # extra penalty on money spent (discourages waste)

REQUIRED_COLUMNS = [
    "distance_km",
    "package_weight",
    "delivery_priority",
    "customer_type",
    "traffic_level",
    "weather_risk",
    "driver_available",
    "driver_rating",
    "delivery_option",
    "estimated_time",
    "delivery_cost",
    "delivery_reward",
    "delivery_success",
]
OPTIONAL_COLUMNS = [
    "package_id",
    "package_type",
    "customer_satisfaction",
    "delay_minutes",
]
MIN_ROWS = 100
MIN_ROWS_PER_OPTION = 15


class DataValidationError(ValueError):
    """Raised when an uploaded / loaded dataset cannot be used."""


# --------------------------------------------------------------------------
# Secrets / environment
# --------------------------------------------------------------------------
def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a setting from environment variables first, then Streamlit secrets."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    try:
        import streamlit as st

        if name in st.secrets:
            secret_value = str(st.secrets[name]).strip()
            if secret_value:
                return secret_value
    except Exception:  # no secrets file, or not running inside Streamlit
        pass
    return default


# --------------------------------------------------------------------------
# Business rules
# --------------------------------------------------------------------------
def delivery_fee(distance_km: float, weight_kg: float, priority: str, customer_type: str) -> float:
    """Money the customer pays us for a delivery (independent of the option chosen)."""
    base = 4.0 + 0.9 * distance_km + 0.35 * weight_kg
    return base * PRIORITY_REVENUE_MULT.get(priority, 1.0) * CUSTOMER_REVENUE_MULT.get(customer_type, 1.0)


def compute_reward(
    *,
    revenue: float,
    cost: float,
    time_min: float,
    success_prob: float,
    priority: str,
    customer_type: str,
    package_type: str = "Parcel",
) -> dict[str, float]:
    """The reward used by the RL agent.

        reward = delivery_profit - cost_penalty - delay_penalty - risk_penalty
                 + customer_satisfaction

    ``success_prob`` is a probability for *expected* rewards and 0 / 1 for a
    *realised* (historical) delivery, so the same formula is used everywhere.
    """
    promised = PROMISED_MINUTES.get(priority, 360)
    delay = max(0.0, time_min - promised)
    profit = success_prob * revenue - cost
    cost_penalty = COST_PENALTY_RATE * cost
    delay_penalty = delay * DELAY_RATE.get(priority, 0.05) * PACKAGE_DELAY_MULT.get(package_type, 1.0)
    risk_penalty = (1.0 - success_prob) * FAILURE_PENALTY
    sat_score = success_prob * max(0.0, 1.0 - 1.2 * delay / promised)  # 0..1
    satisfaction = SATISFACTION_WEIGHT.get(customer_type, 20.0) * sat_score
    reward = profit - cost_penalty - delay_penalty - risk_penalty + satisfaction
    return {
        "profit": float(profit),
        "cost_penalty": float(cost_penalty),
        "delay_penalty": float(delay_penalty),
        "risk_penalty": float(risk_penalty),
        "satisfaction": float(satisfaction),
        "satisfaction_score": float(sat_score),
        "delay_minutes": float(delay),
        "reward": float(reward),
    }


def risk_level(risk_score: float) -> str:
    if risk_score < 0.12:
        return "Low"
    if risk_score < 0.25:
        return "Medium"
    return "High"


def weather_bucket(weather_risk: float) -> int:
    if weather_risk < 0.25:
        return 0  # good
    if weather_risk < 0.60:
        return 1  # rain
    return 2      # storm


def distance_bucket(distance_km: float) -> int:
    if distance_km < 8:
        return 0
    if distance_km < 25:
        return 1
    return 2


def weight_bucket(weight_kg: float) -> int:
    if weight_kg <= 5:
        return 0
    if weight_kg <= 25:
        return 1
    return 2


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def fmt_minutes(minutes: float) -> str:
    minutes = int(round(minutes))
    if minutes < 60:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours}h {rest:02d}m"


# --------------------------------------------------------------------------
# CSV validation
# --------------------------------------------------------------------------
_TRUE = {"1", "true", "yes", "y", "t", "available", "success", "delivered"}
_FALSE = {"0", "false", "no", "n", "f", "unavailable", "failed", "fail"}


def _to_binary(series: pd.Series, column: str) -> pd.Series:
    def convert(value: Any) -> float:
        if isinstance(value, (bool, np.bool_)):
            return 1.0 if value else 0.0
        text = str(value).strip().lower()
        if text in _TRUE:
            return 1.0
        if text in _FALSE:
            return 0.0
        try:
            return 1.0 if float(text) >= 0.5 else 0.0
        except ValueError:
            return np.nan

    out = series.map(convert)
    if out.isna().all():
        raise DataValidationError(f"Column '{column}' must contain 0/1 or yes/no values.")
    return out


def validate_dataframe(df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Check and clean a delivery dataset. Returns (clean_df, warnings)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise DataValidationError("The dataset is empty or missing. Upload a CSV or use the demo data.")

    warnings: list[str] = []
    data = df.copy()
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise DataValidationError(
            "The CSV is missing required columns: "
            + ", ".join(missing)
            + ".  Required columns: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    numeric = ["distance_km", "package_weight", "weather_risk", "driver_rating",
               "estimated_time", "delivery_cost", "delivery_reward"]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["driver_available"] = _to_binary(data["driver_available"], "driver_available")
    data["delivery_success"] = _to_binary(data["delivery_success"], "delivery_success")

    for col, allowed in (
        ("delivery_priority", PRIORITIES),
        ("customer_type", CUSTOMER_TYPES),
        ("traffic_level", TRAFFIC_LEVELS),
    ):
        data[col] = data[col].astype(str).str.strip().str.title()
        bad = ~data[col].isin(allowed)
        if bad.any():
            warnings.append(
                f"Dropped {int(bad.sum())} rows with unknown '{col}' values (allowed: {', '.join(allowed)})."
            )
            data = data[~bad]

    data["delivery_option"] = data["delivery_option"].astype(str).str.strip()
    if "package_type" in data.columns:
        data["package_type"] = data["package_type"].astype(str).str.strip().str.title()
        data.loc[~data["package_type"].isin(PACKAGE_TYPES), "package_type"] = "Parcel"
    else:
        data["package_type"] = "Parcel"

    before = len(data)
    data = data.dropna(subset=numeric + ["driver_available", "delivery_success"])
    if len(data) < before:
        warnings.append(f"Dropped {before - len(data)} rows with missing / non-numeric values.")

    before = len(data)
    data = data[
        (data["distance_km"] > 0)
        & (data["package_weight"] > 0)
        & (data["estimated_time"] > 0)
        & (data["delivery_cost"] >= 0)
        & (data["weather_risk"].between(0, 1))
    ]
    if len(data) < before:
        warnings.append(f"Dropped {before - len(data)} rows with out-of-range values.")
    data["driver_rating"] = data["driver_rating"].clip(1, 5)

    counts = data["delivery_option"].value_counts()
    small = counts[counts < MIN_ROWS_PER_OPTION].index.tolist()
    if small:
        warnings.append(
            "Ignored delivery options with fewer than "
            f"{MIN_ROWS_PER_OPTION} rows: {', '.join(map(str, small))}."
        )
        data = data[~data["delivery_option"].isin(small)]

    if len(data) < MIN_ROWS:
        raise DataValidationError(
            f"Only {len(data)} usable rows after cleaning - at least {MIN_ROWS} are needed to train the RL agent."
        )
    if data["delivery_option"].nunique() < 2:
        raise DataValidationError("The dataset must contain at least 2 different delivery options.")

    return data.reset_index(drop=True), warnings
