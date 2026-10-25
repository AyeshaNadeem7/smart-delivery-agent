"""CrewAI tasks + the plain-text "briefs" that feed them.

The numbers come from the estimator and the RL agent (Python code).  They are
written into the task descriptions as SYSTEM DATA, so the LLM can reason about
them and explain them, but it cannot change or invent them.

NOTE: CrewAI treats {curly_braces} in task text as template variables, so the
briefs below never contain curly braces.
"""
from __future__ import annotations

from typing import Any

from crewai import Agent, Task

from utils import fmt_minutes, fmt_money


def _clean(text: str) -> str:
    return str(text).replace("{", "(").replace("}", ")")


def build_briefs(result: dict[str, Any]) -> dict[str, str]:
    """Turn the decision result into plain-text blocks for the agents."""
    req = result["request"]
    rows = result["options"]
    sel = next(r for r in rows if r["option"] == result["selected"])
    stats = result["rl_stats"]

    request_text = (
        f"Package: {req['package_weight']} kg, type {req.get('package_type', 'Parcel')}\n"
        f"Distance: {req['distance_km']} km | Priority: {req['priority']} | Customer: {req['customer_type']}\n"
        f"Traffic: {req['traffic_level']} | Weather: {req['weather']} (risk {req['weather_risk']:.2f})\n"
        f"Drivers available now: {'yes' if req.get('driver_available', True) else 'no'} | "
        f"Average driver rating: {req.get('driver_rating', 4.3)}\n"
        f"Maximum budget: {fmt_money(req['max_budget'])} | Maximum delivery time: {fmt_minutes(req['max_time_min'])}\n"
        f"Customer promise for this priority: {fmt_minutes(result['promised_min'])}\n"
        f"Options requested: {', '.join(req['options'])}\n"
        f"Customer notes: {_clean(req.get('notes') or 'none')}"
    )

    cost_time_text = "\n".join(
        f"- {r['option']}: cost {fmt_money(r['cost'])}, time {fmt_minutes(r['time_min'])}, "
        f"expected delay {fmt_minutes(r['expected_delay_min'])}, "
        f"{'; '.join(r['violations']) if r['violations'] else 'meets budget and deadline'}"
        for r in rows
    )

    risk_text = "\n".join(
        f"- {r['option']}: success probability {r['p_success'] * 100:.1f}%, "
        f"risk score {r['risk_score']:.2f} ({r['risk_level']}), "
        f"customer satisfaction estimate {r['satisfaction_pct']:.0f}%"
        for r in rows
    )

    rl_options = "\n".join(
        f"- {r['option']}: Q-value {r['q_value']:.2f} (learned from {r['rl_samples_similar']} similar deliveries), "
        f"formula reward {r['formula_reward']:.2f}, status: {r['recommendation']}"
        for r in rows
    )
    rl_text = (
        f"RL ENGINE SELECTED: {result['selected']} (Q-value {sel['q_value']:.2f}).\n"
        f"Training: {stats.get('episodes')} episodes on {stats.get('training_rows')} historical deliveries, "
        f"{stats.get('matched_updates')} learning updates.\n"
        f"Held-out evaluation: average reward {stats.get('policy_avg_reward', float('nan')):.2f} for the learned policy "
        f"versus {stats.get('baseline_avg_reward', float('nan')):.2f} for the historical policy "
        f"({stats.get('reward_improvement_pct', float('nan')):+.1f}%).\n"
        f"Reward formula: profit - cost penalty - delay penalty - risk penalty + customer satisfaction.\n"
        f"Per-option RL values:\n{rl_options}\n"
        f"Reasons computed by the system: {' | '.join(result['reasons'])}\n"
        f"System warnings: {' | '.join(result['warnings']) if result['warnings'] else 'none'}"
    )
    return {
        "request": request_text,
        "cost_time": cost_time_text,
        "risk": risk_text,
        "rl": rl_text,
        "selected": result["selected"],
    }


def create_tasks(agents: dict[str, Agent], briefs: dict[str, str]) -> list[Task]:
    """Five sequential tasks. Later tasks receive earlier outputs through ``context``."""
    t_analyst = Task(
        description=(
            "Analyse this delivery request and write a short brief (max 120 words).\n"
            "Summarise what is shipped, the urgency, the constraints and any special customer notes. "
            "Mention which options look realistic or unrealistic and why.\n\n"
            "SYSTEM DATA - REQUEST\n" + briefs["request"] + "\n\n"
            "SYSTEM DATA - OPTION CHECKS\n" + briefs["cost_time"]
        ),
        expected_output="A brief of at most 120 words summarising the delivery requirements and constraints.",
        agent=agents["analyst"],
    )
    t_cost = Task(
        description=(
            "Compare cost and delivery time of the options (max 150 words). "
            "Use ONLY the figures below. Identify the cheapest, the fastest and the main cost-versus-speed trade-off.\n\n"
            "SYSTEM DATA - COST AND TIME ESTIMATES (regression models)\n" + briefs["cost_time"]
        ),
        expected_output="A comparison of cost versus time with the cheapest and fastest option named (max 150 words).",
        agent=agents["cost_time"],
        context=[t_analyst],
    )
    t_risk = Task(
        description=(
            "Assess delay and failure risk (max 150 words). Use ONLY the figures below. "
            "Explain which options are riskiest and what drives the risk "
            "(traffic, weather, distance, driver availability or rating).\n\n"
            "SYSTEM DATA - RISK ESTIMATES (classifier)\n" + briefs["risk"] + "\n\n"
            "SYSTEM DATA - CONDITIONS\n" + briefs["request"]
        ),
        expected_output="A risk assessment naming the lowest-risk and highest-risk options (max 150 words).",
        agent=agents["risk"],
        context=[t_analyst],
    )
    t_rl = Task(
        description=(
            "Report the decision of the reinforcement learning engine (max 150 words). "
            "State clearly that the RL engine computed the Q-values and made the selection, "
            "then explain in plain language what the numbers mean and how confident the engine is. "
            "Do NOT claim you ran any RL calculation, and do not change any number.\n\n"
            "SYSTEM DATA - RL ENGINE OUTPUT\n" + briefs["rl"]
        ),
        expected_output="A plain-language report of the RL engine's choice, values and confidence (max 150 words).",
        agent=agents["rl"],
        context=[t_cost, t_risk],
    )
    t_manager = Task(
        description=(
            f"Write the final recommendation. You MUST recommend exactly this option: {briefs['selected']} "
            "(it was selected by the RL engine). Use the reports of your colleagues and the system data. "
            "Format in Markdown with: **Recommendation**, **Why this option** (3-5 bullets), "
            "**Trade-offs** and **Things to watch**. Maximum 220 words. Do not invent numbers.\n\n"
            "SYSTEM DATA - RL ENGINE OUTPUT\n" + briefs["rl"]
        ),
        expected_output=(
            f"A Markdown recommendation for {briefs['selected']} with reasons, trade-offs and caveats (max 220 words)."
        ),
        agent=agents["manager"],
        context=[t_analyst, t_cost, t_risk, t_rl],
    )
    return [t_analyst, t_cost, t_risk, t_rl, t_manager]
