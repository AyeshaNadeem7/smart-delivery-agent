"""CrewAI agents + the Groq LLM connection.

Groq exposes an OpenAI-compatible API, so CrewAI's built-in OpenAI provider is
pointed at Groq's endpoint.  No extra packages are needed.
"""
from __future__ import annotations

import os
import tempfile

# Keep CrewAI quiet and make sure it never needs a writable home directory.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_STORAGE_DIR", os.path.join(tempfile.gettempdir(), "smartroute_crewai"))

from crewai import LLM, Agent  # noqa: E402

from utils import get_secret  # noqa: E402

DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class MissingApiKeyError(RuntimeError):
    """GROQ_API_KEY is not configured."""


def get_model_name() -> str:
    model = get_secret("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    return model[len("groq/"):] if model.startswith("groq/") else model


def has_api_key() -> bool:
    return bool(get_secret("GROQ_API_KEY"))


def build_llm(temperature: float = 0.2, max_tokens: int = 700) -> LLM:
    """Create the Groq-backed CrewAI LLM. Raises MissingApiKeyError without a key."""
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise MissingApiKeyError(
            "GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml, Streamlit Cloud secrets "
            "or your environment variables."
        )
    model = get_model_name()
    # CrewAI strips a leading "openai/" provider prefix; Groq's own model ids such as
    # "openai/gpt-oss-120b" contain that prefix, so it is doubled to keep it.
    if "/" in model:
        model = f"openai/{model}"
    return LLM(
        model=model,
        custom_openai=True,
        base_url=get_secret("GROQ_BASE_URL", GROQ_BASE_URL),
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=2,
    )


def create_agents(llm: LLM) -> dict[str, Agent]:
    """The five specialists. They reason about numbers they are GIVEN - they never invent them."""
    common = dict(llm=llm, verbose=False, allow_delegation=False, max_iter=3)
    return {
        "analyst": Agent(
            role="Delivery Analyst",
            goal="Understand the package, the customer's needs and the available delivery options.",
            backstory=(
                "You are a logistics analyst who turns a delivery request into a clear brief: "
                "what is being shipped, how urgent it is, which constraints (budget, deadline, "
                "weight limits) apply and which options are realistic."
            ),
            **common,
        ),
        "cost_time": Agent(
            role="Cost & Time Analyst",
            goal="Compare delivery cost and delivery time for every option and explain the trade-off.",
            backstory=(
                "You are a pricing and scheduling specialist. You only use the cost and time "
                "estimates produced by the system's regression models and never make up figures."
            ),
            **common,
        ),
        "risk": Agent(
            role="Risk & Traffic Analyst",
            goal="Assess delay and failure risk for every option using traffic, weather, distance and driver data.",
            backstory=(
                "You are a risk analyst. You interpret the success probabilities and risk levels "
                "provided by the system's classifier and point out what drives the risk."
            ),
            **common,
        ),
        "rl": Agent(
            role="RL Decision Analyst",
            goal="Report and interpret the decision made by the reinforcement learning engine.",
            backstory=(
                "You work with a reinforcement learning engine (a contextual bandit trained with "
                "Q-learning updates on historical deliveries). The engine - not you - computes "
                "rewards and Q-values. You explain its output faithfully and never claim that you "
                "performed any RL calculation yourself."
            ),
            **common,
        ),
        "manager": Agent(
            role="Delivery Manager",
            goal="Write the final recommendation and explain why the RL-selected option is the best choice.",
            backstory=(
                "You are the delivery operations manager. You read your team's reports and present "
                "a short, honest recommendation to the customer. You always recommend the option "
                "selected by the RL engine and describe trade-offs and caveats clearly."
            ),
            **common,
        ),
    }
