"""Reinforcement learning decision layer (contextual bandit / Q-learning, gamma = 0).

* State  = a discretised version of the delivery situation
* Action = a delivery option
* Reward = profit - cost_penalty - delay_penalty - risk_penalty + satisfaction
          (see ``utils.compute_reward``; for historical rows it is the
          ``delivery_reward`` column)

Training is *offline replay*: for every episode we draw a historical delivery,
let the agent pick an action (epsilon-greedy) and learn from it only when the
agent's action matches what really happened in the data (the standard replay
method for evaluating contextual bandits on logged data).

``recommend`` combines the estimator (cost/time/risk predictions), hard user
constraints (budget, deadline) and the learned Q-values.  The option with the
highest Q-value among the feasible ones is the final decision - the LLM never
chooses.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from estimator import OptionEstimator
from utils import (
    PRIORITIES,
    CUSTOMER_TYPES,
    PROMISED_MINUTES,
    TRAFFIC_CODE,
    WEATHER_CONDITIONS,
    compute_reward,
    delivery_fee,
    distance_bucket,
    risk_level,
    weather_bucket,
    weight_bucket,
)


class NoFeasibleOptionError(ValueError):
    """No selected delivery option can physically carry the package."""


# ==========================================================================
# RL agent
# ==========================================================================
class DeliveryRLAgent:
    """Tabular contextual-bandit agent with hierarchical value estimates."""

    def __init__(self, actions: list[str], max_weight: dict[str, float],
                 alpha_min: float = 0.05, shrink: float = 5.0, seed: int = 42) -> None:
        self.actions = list(actions)
        self.k = len(self.actions)
        self.action_index = {a: i for i, a in enumerate(self.actions)}
        self.max_w = np.array([max_weight.get(a, 1e9) * 1.25 for a in self.actions])
        self.alpha_min = alpha_min
        self.shrink = shrink
        self.seed = seed
        # value tables: root (per action) -> coarse state -> fine state
        self.root_q = np.zeros(self.k)
        self.root_n = np.zeros(self.k)
        self.coarse_q: dict[tuple, np.ndarray] = {}
        self.coarse_n: dict[tuple, np.ndarray] = {}
        self.fine_q: dict[tuple, np.ndarray] = {}
        self.fine_n: dict[tuple, np.ndarray] = {}
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------- states
    @staticmethod
    def state_keys(priority: str, customer: str, traffic: str, weather_risk: float,
                   distance_km: float, weight_kg: float) -> tuple[tuple, tuple]:
        coarse = (PRIORITIES.index(priority), TRAFFIC_CODE[traffic],
                  weather_bucket(weather_risk), distance_bucket(distance_km))
        fine = coarse + (CUSTOMER_TYPES.index(customer), weight_bucket(weight_kg))
        return coarse, fine

    # ------------------------------------------------------------- values
    def _table(self, store: dict, key: tuple) -> np.ndarray:
        if key not in store:
            store[key] = np.zeros(self.k)
        return store[key]

    def q_effective(self, coarse: tuple, fine: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Blend fine -> coarse -> root estimates (more data = more trust in the fine table)."""
        qc, nc = self.coarse_q.get(coarse), self.coarse_n.get(coarse)
        if qc is None:
            qc, nc = np.zeros(self.k), np.zeros(self.k)
        qf, nf = self.fine_q.get(fine), self.fine_n.get(fine)
        if qf is None:
            qf, nf = np.zeros(self.k), np.zeros(self.k)
        s = self.shrink
        qc_eff = (nc * qc + s * self.root_q) / (nc + s)
        qf_eff = (nf * qf + s * qc_eff) / (nf + s)
        return qf_eff, nf, nc

    def greedy_action(self, coarse: tuple, fine: tuple, weight_kg: float) -> int:
        q, _, _ = self.q_effective(coarse, fine)
        feasible = weight_kg <= self.max_w
        if not feasible.any():
            return int(np.argmax(q))
        return int(np.argmax(np.where(feasible, q, -np.inf)))

    def _update(self, coarse: tuple, fine: tuple, a: int, reward: float) -> None:
        for q, n in (
            (self._table(self.fine_q, fine), self._table(self.fine_n, fine)),
            (self._table(self.coarse_q, coarse), self._table(self.coarse_n, coarse)),
        ):
            n[a] += 1
            q[a] += max(1.0 / n[a], self.alpha_min) * (reward - q[a])
        self.root_n[a] += 1
        self.root_q[a] += max(1.0 / self.root_n[a], self.alpha_min) * (reward - self.root_q[a])

    # ----------------------------------------------------------- training
    def _encode(self, df: pd.DataFrame):
        keys, weights = [], []
        for r in df.itertuples(index=False):
            keys.append(self.state_keys(r.delivery_priority, r.customer_type, r.traffic_level,
                                        float(r.weather_risk), float(r.distance_km), float(r.package_weight)))
            weights.append(float(r.package_weight))
        actions = np.array([self.action_index[o] for o in df["delivery_option"]])
        rewards = df["delivery_reward"].to_numpy(float)
        return keys, np.array(weights), actions, rewards

    def fit(self, df: pd.DataFrame, episodes: int = 40000, holdout: float = 0.2) -> dict[str, Any]:
        """Train on ``1 - holdout`` of the data and evaluate on the rest."""
        if df is None or df.empty:
            raise ValueError("Cannot train the RL agent on an empty dataset.")
        episodes = int(max(1000, episodes))
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(df))
        n_test = int(len(df) * holdout)
        test_df = df.iloc[order[:n_test]]
        train_df = df.iloc[order[n_test:]]
        if len(train_df) < 50:
            raise ValueError("Not enough rows to train the RL agent.")

        keys, weights, logged_a, rewards = self._encode(train_df)
        n = len(train_df)
        idx = rng.integers(0, n, episodes)
        u = rng.random(episodes)
        pick = rng.random(episodes)

        eps_start, eps_end = 1.0, 0.05
        decay_until = 0.7 * episodes
        n_blocks = 40
        block = max(1, episodes // n_blocks)
        curve: list[dict[str, float]] = []
        block_sum, block_n = 0.0, 0
        matched_total, best_reward = 0, -np.inf
        last_avg = float(rewards.mean())

        for ep in range(episodes):
            i = idx[ep]
            coarse, fine = keys[i]
            eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_until)
            feasible = np.flatnonzero(weights[i] <= self.max_w)
            if feasible.size == 0:
                feasible = np.arange(self.k)
            if u[ep] < eps:
                a = int(feasible[int(pick[ep] * feasible.size) % feasible.size])
            else:
                a = self.greedy_action(coarse, fine, weights[i])
            if a == logged_a[i]:
                r = float(rewards[i])
                self._update(coarse, fine, a, r)
                block_sum += r
                block_n += 1
                matched_total += 1
                best_reward = max(best_reward, r)
            if (ep + 1) % block == 0:
                if block_n:
                    last_avg = block_sum / block_n
                curve.append({"episode": ep + 1, "avg_reward": last_avg, "epsilon": eps})
                block_sum, block_n = 0.0, 0

        self.stats = {
            "episodes": episodes,
            "training_rows": n,
            "matched_updates": matched_total,
            "curve": curve,
            "best_reward": float(best_reward) if matched_total else float("nan"),
            "final_avg_reward": float(curve[-1]["avg_reward"]) if curve else float("nan"),
            "first_avg_reward": float(curve[0]["avg_reward"]) if curve else float("nan"),
        }
        self.stats.update(self.evaluate(test_df if n_test >= 50 else train_df,
                                        held_out=n_test >= 50))
        return self.stats

    def evaluate(self, df: pd.DataFrame, held_out: bool = True) -> dict[str, Any]:
        """Replay evaluation of the greedy policy against the historical (random-ish) policy."""
        keys, weights, logged_a, rewards = self._encode(df)
        baseline = float(rewards.mean())
        greedy = np.array([self.greedy_action(c, f, w) for (c, f), w in zip(keys, weights)])
        match = greedy == logged_a
        policy_value = float(rewards[match].mean()) if match.any() else float("nan")
        improvement = policy_value - baseline if match.any() else float("nan")
        pct = improvement / abs(baseline) * 100 if match.any() and baseline != 0 else float("nan")
        return {
            "eval_rows": int(len(df)),
            "eval_held_out": held_out,
            "eval_matched": int(match.sum()),
            "baseline_avg_reward": baseline,
            "policy_avg_reward": policy_value,
            "reward_improvement": improvement,
            "reward_improvement_pct": pct,
        }


# ==========================================================================
# Decision layer
# ==========================================================================
def build_system(df: pd.DataFrame, episodes: int = 40000, seed: int = 42) -> tuple[OptionEstimator, DeliveryRLAgent]:
    """Fit the estimator and train the RL agent on a validated dataset."""
    estimator = OptionEstimator().fit(df)
    agent = DeliveryRLAgent(estimator.options, estimator.max_weight, seed=seed)
    agent.fit(df, episodes=episodes)
    return estimator, agent


def _violation_labels(cost: float, time_min: float, budget: float, deadline: float) -> list[str]:
    labels = []
    if cost > budget:
        labels.append("Over budget")
    if time_min > deadline:
        labels.append("Too slow")
    return labels


def recommend(request: dict[str, Any], estimator: OptionEstimator, agent: DeliveryRLAgent) -> dict[str, Any]:
    """Score every available option and let the RL agent choose.

    ``request`` keys: package_weight, distance_km, priority, customer_type,
    traffic_level, weather (name), package_type, options (list), max_budget,
    max_time_min, driver_available, driver_rating, notes.
    """
    weather_risk = WEATHER_CONDITIONS[request["weather"]]
    ctx = {
        "distance_km": float(request["distance_km"]),
        "package_weight": float(request["package_weight"]),
        "traffic_level": request["traffic_level"],
        "weather_risk": weather_risk,
        "driver_available": bool(request.get("driver_available", True)),
        "driver_rating": float(request.get("driver_rating", 4.3)),
    }
    priority, customer = request["priority"], request["customer_type"]
    ptype = request.get("package_type", "Parcel")
    budget, deadline = float(request["max_budget"]), float(request["max_time_min"])
    promised = PROMISED_MINUTES[priority]
    revenue = delivery_fee(ctx["distance_km"], ctx["package_weight"], priority, customer)

    coarse, fine = agent.state_keys(priority, customer, ctx["traffic_level"], weather_risk,
                                    ctx["distance_km"], ctx["package_weight"])
    q_all, n_fine, n_coarse = agent.q_effective(coarse, fine)

    selected_options = [o for o in request.get("options", []) if o in agent.action_index]
    if not selected_options:
        raise NoFeasibleOptionError("None of the chosen delivery options exist in the dataset.")

    rows: list[dict[str, Any]] = []
    for option in selected_options:
        est = estimator.estimate(ctx, option)
        parts = compute_reward(revenue=revenue, cost=est["cost"], time_min=est["time_min"],
                               success_prob=est["p_success"], priority=priority,
                               customer_type=customer, package_type=ptype)
        on_time = 1.0 - min(1.0, parts["delay_minutes"] / promised)
        risk_score = 1.0 - est["p_success"] * on_time
        a = agent.action_index[option]
        weight_ok = estimator.is_feasible_weight(option, ctx["package_weight"])
        violations = _violation_labels(est["cost"], est["time_min"], budget, deadline)
        if not weight_ok:
            violations = ["Weight limit"]
        rows.append({
            "option": option,
            "cost": est["cost"],
            "time_min": est["time_min"],
            "p_success": est["p_success"],
            "risk_score": risk_score,
            "risk_level": risk_level(risk_score),
            "expected_delay_min": parts["delay_minutes"],
            "formula_reward": parts["reward"],
            "reward_parts": {k: parts[k] for k in
                             ("profit", "cost_penalty", "delay_penalty", "risk_penalty", "satisfaction")},
            "satisfaction_pct": 100.0 * parts["satisfaction_score"],
            "q_value": float(q_all[a]),
            "rl_samples_similar": int(n_fine[a]),
            "rl_samples_coarse": int(n_coarse[a]),
            "weight_ok": weight_ok,
            "violations": violations,
            "feasible": not violations,
        })

    warnings: list[str] = []
    carriable = [r for r in rows if r["weight_ok"]]
    if not carriable:
        raise NoFeasibleOptionError(
            "No selected delivery option can carry a package of this weight. Select a heavier-capacity option."
        )

    candidates = [r for r in rows if r["feasible"]]
    relaxed = False
    if not candidates:
        relaxed = True

        def excess(r: dict) -> float:
            return max(0.0, r["cost"] / budget - 1) + max(0.0, r["time_min"] / deadline - 1)

        best_violation = min(carriable, key=excess)
        candidates = [best_violation]
        warnings.append(
            "No option satisfies both your budget and time limit. Showing the closest match - "
            "consider raising the budget or relaxing the deadline."
        )

    selected = max(candidates, key=lambda r: r["q_value"])

    qs = np.array([r["q_value"] for r in candidates])
    span = qs.max() - qs.min()
    for r in rows:
        if r in candidates and not relaxed:
            norm = (r["q_value"] - qs.min()) / span if span > 1e-9 else 1.0
            r["score"] = round(40 + 60 * float(norm), 1)
        elif r is selected:
            r["score"] = 40.0
        else:
            r["score"] = None
        if r is selected:
            r["recommendation"] = "Closest match" if relaxed else "Recommended"
        elif r["feasible"]:
            r["recommendation"] = "Alternative"
        else:
            r["recommendation"] = "Excluded: " + " & ".join(r["violations"])

    # Human-readable reasons (computed by code, not by the LLM)
    reasons: list[str] = []
    pool = candidates
    if relaxed:
        reasons.append("Closest to your limits - no option met both the budget and the time limit.")
    else:
        reasons.append(
            f"Highest learned RL value (Q = {selected['q_value']:.2f}) among {len(pool)} feasible option(s), "
            f"based on {selected['rl_samples_similar']} similar historical deliveries "
            f"({selected['rl_samples_coarse']} in the wider group)."
        )
    if selected["cost"] <= min(r["cost"] for r in pool) + 1e-9 and len(pool) > 1:
        reasons.append("Lowest expected cost among the feasible options.")
    elif selected["cost"] <= budget:
        reasons.append(f"Within budget (${selected['cost']:.2f} of ${budget:.2f}).")
    if selected["time_min"] <= min(r["time_min"] for r in pool) + 1e-9 and len(pool) > 1:
        reasons.append("Fastest acceptable delivery time.")
    elif selected["time_min"] <= deadline:
        reasons.append(f"Meets your deadline ({selected['time_min']:.0f} of {deadline:.0f} minutes).")
    if selected["p_success"] >= max(r["p_success"] for r in pool) - 0.005 or selected["p_success"] >= 0.93:
        reasons.append(f"High reliability ({selected['p_success'] * 100:.1f}% estimated success probability).")
    if selected["formula_reward"] >= max(r["formula_reward"] for r in pool) - 1e-9 and len(pool) > 1:
        reasons.append("Highest expected reward by the reward formula.")
    if selected["risk_level"] == "Low":
        reasons.append("Low delay / failure risk.")
    elif selected["risk_level"] == "High":
        warnings.append("The selected option still carries a high delay / failure risk.")
    if selected["rl_samples_similar"] < 5:
        warnings.append("The RL agent has seen few similar deliveries - its confidence for this case is limited.")

    return {
        "request": {**request, "weather_risk": weather_risk, "package_type": ptype},
        "revenue": revenue,
        "promised_min": promised,
        "options": rows,
        "selected": selected["option"],
        "relaxed": relaxed,
        "reasons": reasons,
        "warnings": warnings,
        "rl_stats": copy.deepcopy(agent.stats),
        "model_metrics": dict(estimator.metrics),
    }


def what_if_scenarios(request: dict[str, Any], estimator: OptionEstimator, agent: DeliveryRLAgent) -> list[dict[str, Any]]:
    """Re-run the decision layer for a few changed inputs (used by the chat assistant)."""
    variants: list[tuple[str, dict[str, Any]]] = []
    for factor in (1.25, 1.5, 2.0):
        variants.append((f"Budget x{factor} (${request['max_budget'] * factor:.2f})",
                         {**request, "max_budget": request["max_budget"] * factor}))
    for minutes in (60, 120, 240):
        if minutes != request["max_time_min"]:
            variants.append((f"Deadline {minutes} minutes", {**request, "max_time_min": float(minutes)}))
    for traffic in ("Low", "High"):
        if traffic != request["traffic_level"]:
            variants.append((f"Traffic {traffic}", {**request, "traffic_level": traffic}))
    for priority in ("Standard", "Urgent"):
        if priority != request["priority"]:
            variants.append((f"Priority {priority}", {**request, "priority": priority}))

    out = []
    for label, changed in variants:
        try:
            res = recommend(changed, estimator, agent)
            sel = next(r for r in res["options"] if r["option"] == res["selected"])
            out.append({"scenario": label, "selected": sel["option"], "cost": round(sel["cost"], 2),
                        "time_min": round(sel["time_min"]), "q_value": round(sel["q_value"], 2),
                        "risk_level": sel["risk_level"], "constraints_met": not res["relaxed"]})
        except NoFeasibleOptionError as exc:
            out.append({"scenario": label, "selected": None, "note": str(exc)})
    return out
