"""Synthetic delivery data.

This file simulates the "world": five delivery options with different speed,
price, reliability and weather sensitivity.  It produces *historical
deliveries* (what happened when option X was used in situation Y).  The
estimator and the RL agent never see these hidden profiles - they only learn
from the generated rows, exactly as they would from a real CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    CUSTOMER_TYPES,
    PACKAGE_TYPES,
    PRIORITIES,
    TRAFFIC_FACTOR,
    TRAFFIC_LEVELS,
    compute_reward,
    delivery_fee,
)

# Hidden "ground truth" of the simulated courier network.
OPTION_PROFILES = {
    "Economy Bike": dict(speed=18, base=2.0, per_km=0.35, per_kg=0.15, max_w=10,
                         rel=0.90, weather_s=0.50, traffic_s=0.30, wait=15),
    "Standard Van": dict(speed=32, base=4.0, per_km=0.60, per_kg=0.10, max_w=300,
                         rel=0.93, weather_s=0.25, traffic_s=0.60, wait=45),
    "Express Van": dict(speed=42, base=7.0, per_km=0.95, per_kg=0.14, max_w=300,
                        rel=0.95, weather_s=0.20, traffic_s=0.50, wait=10),
    "Same-Day Priority": dict(speed=38, base=9.0, per_km=1.10, per_kg=0.20, max_w=25,
                              rel=0.94, weather_s=0.45, traffic_s=0.25, wait=8),
    "Scheduled Freight": dict(speed=25, base=3.0, per_km=0.40, per_kg=0.05, max_w=2000,
                              rel=0.96, weather_s=0.10, traffic_s=0.70, wait=120),
}


def _sample_package(rng: np.random.Generator) -> tuple[str, float]:
    ptype = rng.choice(PACKAGE_TYPES, p=[0.20, 0.40, 0.15, 0.15, 0.10])
    if ptype == "Documents":
        weight = rng.uniform(0.1, 1.0)
    elif ptype == "Parcel":
        weight = float(np.clip(rng.lognormal(np.log(4.0), 0.6), 0.5, 25))
    elif ptype == "Fragile":
        weight = rng.uniform(0.5, 8.0)
    elif ptype == "Perishable":
        weight = rng.uniform(1.0, 15.0)
    else:
        weight = rng.uniform(20.0, 120.0)
    return str(ptype), float(round(weight, 2))


def generate_delivery_data(n_rows: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Create ``n_rows`` historical deliveries (one row = one delivery)."""
    rng = np.random.default_rng(seed)
    options = list(OPTION_PROFILES)
    weather_choices = np.array([0.05, 0.15, 0.40, 0.80])
    rows = []

    for i in range(int(n_rows)):
        ptype, weight = _sample_package(rng)
        distance = float(np.clip(rng.gamma(2.0, 8.0), 0.5, 80.0))
        priority = str(rng.choice(PRIORITIES, p=[0.20, 0.45, 0.22, 0.13]))
        customer = str(rng.choice(CUSTOMER_TYPES, p=[0.60, 0.20, 0.20]))
        traffic = str(rng.choice(TRAFFIC_LEVELS, p=[0.30, 0.40, 0.30]))
        weather_risk = float(np.clip(
            rng.choice(weather_choices, p=[0.45, 0.25, 0.20, 0.10]) + rng.normal(0, 0.03), 0.0, 1.0))
        driver_available = int(rng.random() < 0.80)
        driver_rating = float(np.clip(rng.normal(4.3, 0.35), 3.0, 5.0))

        # Historical dispatchers used a mostly random policy -> good coverage
        # of every (situation, option) pair, which is what RL needs to learn.
        feasible = [o for o in options if weight <= OPTION_PROFILES[o]["max_w"]]
        option = str(rng.choice(feasible))
        p = OPTION_PROFILES[option]

        tf = TRAFFIC_FACTOR[traffic]
        travel = distance / p["speed"] * 60.0 * (1 + p["traffic_s"] * tf + p["weather_s"] * weather_risk)
        travel *= float(rng.lognormal(0.0, 0.12))
        time_min = p["wait"] + (0 if driver_available else 30) + travel
        cost = (p["base"] + p["per_km"] * distance + p["per_kg"] * weight)
        cost *= (1 + 0.15 * tf) * (1 + 0.20 * weather_risk) * float(rng.uniform(0.95, 1.05))

        p_success = (
            p["rel"]
            * (1 - 0.5 * weather_risk * p["weather_s"])
            * (1 - 0.1 * tf * p["traffic_s"])
            * (0.9 + 0.1 * (driver_rating - 3.0) / 2.0)
        )
        p_success = float(np.clip(p_success, 0.30, 0.995))
        success = int(rng.random() < p_success)
        if not success:
            time_min *= 1.8  # re-delivery attempt

        revenue = delivery_fee(distance, weight, priority, customer)
        parts = compute_reward(
            revenue=revenue, cost=cost, time_min=time_min, success_prob=float(success),
            priority=priority, customer_type=customer, package_type=ptype,
        )
        rows.append(
            {
                "package_id": f"PKG{i + 1:06d}",
                "distance_km": round(distance, 2),
                "package_weight": weight,
                "package_type": ptype,
                "delivery_priority": priority,
                "customer_type": customer,
                "traffic_level": traffic,
                "weather_risk": round(weather_risk, 3),
                "driver_available": driver_available,
                "driver_rating": round(driver_rating, 2),
                "delivery_option": option,
                "estimated_time": round(time_min, 1),
                "delivery_cost": round(cost, 2),
                "delivery_success": success,
                "delay_minutes": round(parts["delay_minutes"], 1),
                "customer_satisfaction": round(100 * parts["satisfaction_score"], 1),
                "delivery_reward": round(parts["reward"], 3),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # quick manual run: python data_generator.py
    frame = generate_delivery_data(1500, seed=7)
    frame.to_csv("sample_data/delivery_sample.csv", index=False)
    print(frame.head())
    print(frame.groupby("delivery_option")["delivery_reward"].mean())
