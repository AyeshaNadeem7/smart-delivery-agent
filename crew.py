"""Crew assembly, execution, the chat assistant and offline fallbacks."""
from __future__ import annotations

import json
import re
from typing import Any

from agents import MissingApiKeyError, build_llm, create_agents
from tasks import build_briefs, create_tasks
from utils import fmt_minutes, fmt_money

from crewai import Crew, Process  # noqa: E402

AGENT_TITLES = [
    "Delivery Analyst",
    "Cost & Time Analyst",
    "Risk & Traffic Analyst",
    "RL Decision Analyst",
    "Delivery Manager (final recommendation)",
]


# --------------------------------------------------------------------------
# Crew
# --------------------------------------------------------------------------
def run_crew(result: dict[str, Any]) -> dict[str, Any]:
    """Run the 5-agent sequential crew. Raises MissingApiKeyError or any CrewAI/LLM error."""
    llm = build_llm()
    agents = create_agents(llm)
    tasks = create_tasks(agents, build_briefs(result))
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        memory=False,
    )
    output = crew.kickoff()
    reports = []
    for title, task_out in zip(AGENT_TITLES, output.tasks_output):
        reports.append({"agent": title, "text": str(task_out.raw).strip()})
    final = str(output.raw).strip() if getattr(output, "raw", None) else (reports[-1]["text"] if reports else "")
    if not final:
        raise RuntimeError("The crew returned an empty answer.")
    return {"reports": reports, "final": final}


def fallback_report(result: dict[str, Any]) -> dict[str, Any]:
    """Template-based summary used when the LLM is unavailable (clearly labelled in the UI)."""
    sel = next(r for r in result["options"] if r["option"] == result["selected"])
    cheapest = min(result["options"], key=lambda r: r["cost"])
    fastest = min(result["options"], key=lambda r: r["time_min"])
    lines = [
        f"**Recommendation:** {sel['option']}",
        "",
        "**Why this option**",
        *[f"- {reason}" for reason in result["reasons"]],
        "",
        "**Trade-offs**",
        f"- Cheapest option overall: {cheapest['option']} ({fmt_money(cheapest['cost'])}).",
        f"- Fastest option overall: {fastest['option']} ({fmt_minutes(fastest['time_min'])}).",
    ]
    if result["warnings"]:
        lines += ["", "**Things to watch**", *[f"- {w}" for w in result["warnings"]]]
    return {"reports": [], "final": "\n".join(lines)}


# --------------------------------------------------------------------------
# Chat assistant
# --------------------------------------------------------------------------
def _chat_context(result: dict[str, Any], scenarios: list[dict[str, Any]]) -> str:
    req = result["request"]
    options = [
        {
            "option": r["option"],
            "cost": round(r["cost"], 2),
            "time_min": round(r["time_min"]),
            "success_probability": round(r["p_success"], 3),
            "risk_level": r["risk_level"],
            "rl_q_value": round(r["q_value"], 2),
            "formula_reward": round(r["formula_reward"], 2),
            "satisfaction_pct": round(r["satisfaction_pct"]),
            "status": r["recommendation"],
        }
        for r in result["options"]
    ]
    stats = result["rl_stats"]
    payload = {
        "request": {k: req[k] for k in ("package_weight", "distance_km", "priority", "customer_type",
                                        "traffic_level", "weather", "max_budget", "max_time_min")},
        "rl_selected_option": result["selected"],
        "reasons": result["reasons"],
        "warnings": result["warnings"],
        "options": options,
        "rl_training": {
            "episodes": stats.get("episodes"),
            "policy_avg_reward": round(stats.get("policy_avg_reward", 0.0), 2),
            "baseline_avg_reward": round(stats.get("baseline_avg_reward", 0.0), 2),
        },
        "what_if_scenarios_computed_by_rl_engine": scenarios,
    }
    return json.dumps(payload, indent=1)


CHAT_SYSTEM = (
    "You are the SmartRoute AI delivery assistant. A reinforcement learning (RL) engine and regression models "
    "already produced the numbers in the JSON below; you did not compute them and must never claim to have run "
    "RL yourself. Answer the user's question using ONLY that JSON. For 'what if' questions, use the entries in "
    "what_if_scenarios_computed_by_rl_engine; if the scenario is not listed, say so and suggest changing the form "
    "values and re-running the optimisation. Be concise (under 150 words), friendly and specific, quote option "
    "names and figures exactly.\n\nJSON DATA:\n"
)


def chat_answer(question: str, result: dict[str, Any], scenarios: list[dict[str, Any]],
                history: list[dict[str, str]]) -> str:
    """Ask Groq. Raises MissingApiKeyError or an LLM error."""
    llm = build_llm(temperature=0.3, max_tokens=500)
    messages = [{"role": "system", "content": CHAT_SYSTEM + _chat_context(result, scenarios)}]
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})
    answer = llm.call(messages)
    return str(answer).strip()


def rule_based_answer(question: str, result: dict[str, Any], scenarios: list[dict[str, Any]]) -> str:
    """Offline answers for the common questions when no LLM is available."""
    q = question.lower()
    rows = result["options"]
    sel = next(r for r in rows if r["option"] == result["selected"])
    feasible = [r for r in rows if r["weight_ok"]]

    def describe(r: dict) -> str:
        return (f"**{r['option']}** - {fmt_money(r['cost'])}, {fmt_minutes(r['time_min'])}, "
                f"risk {r['risk_level']}, RL value {r['q_value']:.2f}")

    if "cheap" in q or "lowest cost" in q:
        return "Cheapest option: " + describe(min(feasible, key=lambda r: r["cost"]))
    if "fast" in q or "quick" in q:
        return "Fastest option: " + describe(min(feasible, key=lambda r: r["time_min"]))
    if "reward" in q or "profit" in q:
        return ("Highest expected reward by the reward formula: " + describe(max(feasible, key=lambda r: r["formula_reward"]))
                + f"\n\nHighest RL value: {describe(max(feasible, key=lambda r: r['q_value']))}")
    if "budget" in q or "hour" in q or "within" in q or "if " in q:
        lines = [f"- {s['scenario']}: " + (f"{s['selected']} ({fmt_money(s['cost'])}, {fmt_minutes(s['time_min'])})"
                                          if s.get("selected") else "no feasible option") for s in scenarios]
        return "What-if results from the RL engine:\n" + "\n".join(lines)
    return (f"The RL engine selected {describe(sel)}.\n\n" + "\n".join(f"- {r}" for r in result["reasons"]))


def friendly_error(exc: Exception) -> str:
    """Turn technical exceptions into short, safe messages (never includes secrets)."""
    if isinstance(exc, MissingApiKeyError):
        return "No Groq API key found. Add GROQ_API_KEY to your Streamlit secrets or environment."
    text = str(exc)
    text = re.sub(r"gsk_[A-Za-z0-9]+", "gsk_***", text)
    lowered = text.lower()
    if "401" in lowered or "invalid api key" in lowered or "authentication" in lowered:
        return "Groq rejected the API key (authentication error). Check GROQ_API_KEY."
    if "429" in lowered or "rate limit" in lowered:
        return "Groq rate limit reached (free tier). Wait a minute and try again, or set a smaller GROQ_MODEL."
    if "model" in lowered and ("not found" in lowered or "decommission" in lowered or "does not exist" in lowered):
        return "The configured GROQ_MODEL is not available. Check console.groq.com/docs/models and update GROQ_MODEL."
    if "timeout" in lowered or "timed out" in lowered or "connection" in lowered:
        return "Could not reach Groq (network problem or timeout). Please try again."
    return f"The AI service returned an error: {text[:200]}"
