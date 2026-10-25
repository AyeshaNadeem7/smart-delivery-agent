"""SmartRoute AI - RL + LLM Delivery Optimization Agent (Streamlit app).

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents import get_model_name, has_api_key
from crew import (
    chat_answer,
    fallback_report,
    friendly_error,
    rule_based_answer,
    run_crew,
)
from data_generator import generate_delivery_data
from rl_agent import NoFeasibleOptionError, build_system, recommend, what_if_scenarios
from utils import (
    CUSTOMER_TYPES,
    DataValidationError,
    PACKAGE_TYPES,
    PRIORITIES,
    REQUIRED_COLUMNS,
    TRAFFIC_LEVELS,
    WEATHER_CONDITIONS,
    fmt_minutes,
    fmt_money,
    validate_dataframe,
)

st.set_page_config(page_title="SmartRoute AI", page_icon="🚚", layout="wide")

# --------------------------------------------------------------------------
# Styling (light, modern)
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg, #eef2ff 0%, #f8f4ff 45%, #effbf7 100%); }
[data-testid="stSidebar"] { background: #ffffffcc; border-right: 1px solid #e6e9fb; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
.hero { background: linear-gradient(120deg, #6C5CE7 0%, #4C8DFF 55%, #19C3A6 100%);
        border-radius: 22px; padding: 30px 36px; color: #fff; margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(76,92,231,.25); }
.hero h1 { color: #fff; margin: 0 0 6px 0; font-size: 2.1rem; }
.hero p { margin: 0; font-size: 1.02rem; opacity: .95; }
.pill { display:inline-block; background:#ffffff33; border-radius:999px; padding:3px 12px;
        margin: 10px 6px 0 0; font-size:.8rem; }
.card { background:#fff; border-radius:16px; padding:16px 18px; border:1px solid #e8ebfb;
        box-shadow:0 4px 14px rgba(70,80,160,.07); height:100%; }
.card .lbl { font-size:.74rem; color:#7a80a8; text-transform:uppercase; letter-spacing:.05em; }
.card .val { font-size:1.55rem; font-weight:700; color:#262b55; line-height:1.25; }
.card .sub { font-size:.78rem; color:#8b91b5; }
.winner { background: linear-gradient(120deg,#ffffff,#f1eeff); border:2px solid #6C5CE7;
          border-radius:18px; padding:18px 24px; margin-bottom:14px; }
.winner .tag { color:#6C5CE7; font-weight:700; font-size:.8rem; letter-spacing:.08em; }
.winner .name { font-size:2rem; font-weight:800; color:#262b55; }
.section-title { font-size:1.25rem; font-weight:700; color:#262b55; margin:26px 0 8px 0; }
</style>
""",
    unsafe_allow_html=True,
)


def card(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data + training (cached so the page stays fast)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_demo(n_rows: int, seed: int) -> pd.DataFrame:
    return generate_delivery_data(n_rows, seed)


@st.cache_data(show_spinner=False)
def parse_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


@st.cache_resource(show_spinner=False)
def train_cached(data_key: str, _df: pd.DataFrame, episodes: int, seed: int):
    return build_system(_df, episodes=episodes, seed=seed)


def get_dataset(source: str, uploaded, demo_rows: int, seed: int):
    """Return (clean_df, data_key, warnings, error_message)."""
    try:
        if source == "Upload CSV":
            if uploaded is None:
                return None, "", [], "No CSV uploaded yet. Upload a file in the sidebar or switch to the demo data."
            raw = uploaded.getvalue()
            df, warns = validate_dataframe(parse_csv(raw))
            return df, "csv-" + hashlib.md5(raw).hexdigest(), warns, ""
        df, warns = validate_dataframe(load_demo(demo_rows, seed))
        return df, f"demo-{demo_rows}-{seed}", warns, ""
    except DataValidationError as exc:
        return None, "", [], str(exc)
    except Exception as exc:  # unreadable file etc.
        return None, "", [], f"Could not read the dataset: {exc}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    source = st.radio("Delivery data", ["Built-in demo data", "Upload CSV"], index=0)
    uploaded_file = None
    demo_rows = 8000
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload delivery history (CSV)", type=["csv"])
        st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))
    else:
        demo_rows = st.slider("Synthetic deliveries", 2000, 20000, 8000, 1000)
    episodes = st.slider("RL training episodes", 10000, 100000, 40000, 5000)
    seed = st.number_input("Random seed", 0, 9999, 42)

    sample_csv = generate_delivery_data(400, 7).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download sample CSV format", sample_csv, "delivery_sample.csv", "text/csv")

    st.markdown("---")
    if has_api_key():
        st.success(f"Groq connected · model `{get_model_name()}`")
    else:
        st.warning("GROQ_API_KEY not found. The RL engine still works; AI explanations use offline templates.")

dataset, data_key, data_warnings, data_error = get_dataset(source, uploaded_file, demo_rows, int(seed))

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🚚 AI Delivery Optimization Agent</h1>
  <p>SmartRoute AI combines a <b>reinforcement learning</b> decision engine with a <b>CrewAI</b> team of
  LLM agents (Groq). The RL engine decides which delivery option earns the best overall reward; the agents
  analyse the numbers and explain the decision in plain language.</p>
  <span class="pill">Reinforcement Learning decides</span>
  <span class="pill">CrewAI orchestrates</span>
  <span class="pill">Groq LLM explains</span>
</div>
""",
    unsafe_allow_html=True,
)

if data_error:
    st.error(data_error)
for warn in data_warnings:
    st.warning(warn)

if dataset is not None:
    with st.expander(f"📦 Training data: {len(dataset):,} deliveries · {dataset['delivery_option'].nunique()} options"):
        st.dataframe(dataset.head(15), width="stretch", hide_index=True)

option_list = sorted(dataset["delivery_option"].unique().tolist()) if dataset is not None else []

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
section("📝 Delivery request")
with st.form("delivery_form"):
    c1, c2, c3 = st.columns(3)
    weight = c1.number_input("Package weight (kg)", 0.1, 2000.0, 3.0, 0.5)
    distance = c2.number_input("Distance (km)", 0.5, 500.0, 12.0, 0.5)
    package_type = c3.selectbox("Package type", PACKAGE_TYPES, index=1)
    priority = c1.selectbox("Delivery priority", PRIORITIES, index=1)
    customer_type = c2.selectbox("Customer type", CUSTOMER_TYPES, index=0)
    traffic = c3.selectbox("Traffic level", TRAFFIC_LEVELS, index=1)
    weather = c1.selectbox("Weather condition", list(WEATHER_CONDITIONS), index=0)
    max_budget = c2.number_input("Maximum budget ($)", 1.0, 5000.0, 40.0, 1.0)
    max_time = c3.number_input("Maximum delivery time (minutes)", 10, 4320, 240, 10)
    options = st.multiselect("Available delivery options", option_list, default=option_list)
    d1, d2 = st.columns(2)
    driver_available = d1.checkbox("Driver immediately available", value=True)
    driver_rating = d2.slider("Average driver rating", 3.0, 5.0, 4.3, 0.1)
    notes = st.text_area("Special instructions (optional)", placeholder="e.g. Customer needs it before a 5pm meeting.",
                         max_chars=300)
    submitted = st.form_submit_button("🚀 Optimize Delivery", type="primary")

# --------------------------------------------------------------------------
# Run optimisation
# --------------------------------------------------------------------------
if submitted:
    if dataset is None:
        st.error("Cannot optimise without a valid dataset. " + (data_error or ""))
    elif not options:
        st.error("Please select at least one delivery option.")
    else:
        request = {
            "package_weight": float(weight), "distance_km": float(distance), "priority": priority,
            "customer_type": customer_type, "traffic_level": traffic, "weather": weather,
            "package_type": package_type, "options": options, "max_budget": float(max_budget),
            "max_time_min": float(max_time), "driver_available": bool(driver_available),
            "driver_rating": float(driver_rating), "notes": notes.strip(),
        }
        try:
            with st.spinner("Training the reinforcement learning agent (cached after the first run)..."):
                estimator, agent = train_cached(data_key, dataset, int(episodes), int(seed))
            result = recommend(request, estimator, agent)
            scenarios = what_if_scenarios(request, estimator, agent)
        except NoFeasibleOptionError as exc:
            st.error(str(exc))
            result = None
        except Exception as exc:
            st.error(f"The RL decision layer failed: {exc}")
            result = None

        if result is not None:
            crew_out, crew_note, crew_ai = None, "", False
            if has_api_key():
                try:
                    with st.spinner("The CrewAI team is analysing the options (5 agents)..."):
                        crew_out = run_crew(result)
                    crew_ai = True
                except Exception as exc:
                    crew_note = friendly_error(exc)
            else:
                crew_note = "No Groq API key configured - showing an offline summary instead of AI agent reports."
            if crew_out is None:
                crew_out = fallback_report(result)
            st.session_state.update(
                result=result, scenarios=scenarios, crew_out=crew_out, crew_note=crew_note,
                crew_ai=crew_ai, chat=[],
            )

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.get("result")
if result:
    rows = result["options"]
    sel = next(r for r in rows if r["option"] == result["selected"])
    stats = result["rl_stats"]

    section("🏆 Recommendation")
    tag = "CLOSEST MATCH TO YOUR LIMITS" if result["relaxed"] else "RECOMMENDED BY THE RL ENGINE"
    st.markdown(f'<div class="winner"><div class="tag">{tag}</div><div class="name">{sel["option"]}</div></div>',
                unsafe_allow_html=True)
    m = st.columns(6)
    m[0].markdown(card("Estimated cost", fmt_money(sel["cost"]), f"budget {fmt_money(result['request']['max_budget'])}"),
                  unsafe_allow_html=True)
    m[1].markdown(card("Delivery time", fmt_minutes(sel["time_min"]),
                       f"limit {fmt_minutes(result['request']['max_time_min'])}"), unsafe_allow_html=True)
    m[2].markdown(card("Expected reward", f"{sel['q_value']:.1f}", f"formula estimate {sel['formula_reward']:.1f}"),
                  unsafe_allow_html=True)
    m[3].markdown(card("Risk level", sel["risk_level"], f"success {sel['p_success'] * 100:.1f}%"),
                  unsafe_allow_html=True)
    m[4].markdown(card("Customer satisfaction", f"{sel['satisfaction_pct']:.0f}%", "estimated"),
                  unsafe_allow_html=True)
    m[5].markdown(card("Overall score", "-" if sel["score"] is None else f"{sel['score']:.0f}/100", "relative to options"),
                  unsafe_allow_html=True)

    for warn in result["warnings"]:
        st.warning(warn)

    section("💡 Why was this option selected?")
    for reason in result["reasons"]:
        st.markdown(f"✅ {reason}")
    st.caption("These reasons are computed by code from the model outputs - not generated by the LLM.")

    section("🤖 CrewAI agent report")
    crew_out = st.session_state["crew_out"]
    if st.session_state.get("crew_note"):
        st.info(st.session_state["crew_note"])
    st.markdown(crew_out["final"])
    if st.session_state.get("crew_ai") and crew_out["reports"]:
        st.caption("Written by the Delivery Manager agent (Groq LLM) from the reports below. "
                   "All numbers come from the regression models and the RL engine.")
        for rep in crew_out["reports"][:-1]:
            with st.expander(rep["agent"]):
                st.markdown(rep["text"])

    section("📊 Option comparison")
    table = pd.DataFrame({
        "Option": [r["option"] for r in rows],
        "Cost ($)": [round(r["cost"], 2) for r in rows],
        "Time (min)": [round(r["time_min"]) for r in rows],
        "Risk": [f"{r['risk_level']} ({r['risk_score']:.2f})" for r in rows],
        "Reward (RL)": [round(r["q_value"], 2) for r in rows],
        "Score": [r["score"] for r in rows],
        "Recommendation": [("✅ " if r["option"] == result["selected"] else "") + r["recommendation"] for r in rows],
    }).sort_values("Reward (RL)", ascending=False)
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Cost ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Time (min)": st.column_config.NumberColumn(format="%d"),
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        },
    )

    section("🧠 Reinforcement learning")
    r1 = st.columns(5)
    r1[0].markdown(card("Training episodes", f"{stats['episodes']:,}", f"{stats['matched_updates']:,} learning updates"),
                   unsafe_allow_html=True)
    r1[1].markdown(card("Average reward", f"{stats['final_avg_reward']:.1f}", "end of training"), unsafe_allow_html=True)
    r1[2].markdown(card("Best reward", f"{stats['best_reward']:.1f}", "single best delivery seen"), unsafe_allow_html=True)
    r1[3].markdown(card("Selected action", sel["option"], "chosen by max Q-value"), unsafe_allow_html=True)
    imp = stats.get("reward_improvement_pct")
    imp_txt = "n/a" if imp is None or pd.isna(imp) else f"{imp:+.0f}%"
    r1[4].markdown(card("Reward improvement", imp_txt,
                        f"{stats['policy_avg_reward']:.1f} vs {stats['baseline_avg_reward']:.1f} baseline"),
                   unsafe_allow_html=True)
    held = "held-out" if stats.get("eval_held_out") else "training"
    st.caption(f"Improvement = average reward of the learned policy vs. the historical (mostly random) policy on "
               f"{stats['eval_matched']} matching {held} deliveries (replay evaluation).")

    curve = pd.DataFrame(stats["curve"])
    fig_rl = go.Figure()
    fig_rl.add_trace(go.Scatter(x=curve["episode"], y=curve["avg_reward"], mode="lines+markers",
                                name="Average reward", line=dict(color="#6C5CE7", width=3)))
    fig_rl.add_hline(y=stats["baseline_avg_reward"], line_dash="dash", line_color="#19C3A6",
                     annotation_text="historical baseline")
    fig_rl.update_layout(title="RL reward progression", xaxis_title="Training episode",
                         yaxis_title="Average reward", template="plotly_white", height=340,
                         margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_rl, width="stretch")

    section("📈 Visual comparison")
    names = [r["option"] for r in rows]
    colors = ["#6C5CE7" if n == result["selected"] else "#A9B6FF" for n in names]
    v1, v2 = st.columns(2)

    fig_ct = go.Figure(go.Scatter(
        x=[r["time_min"] for r in rows], y=[r["cost"] for r in rows], mode="markers+text", text=names,
        textposition="top center", marker=dict(size=[max(14, min(46, 14 + abs(r["q_value"]))) for r in rows],
                                               color=colors, line=dict(width=1, color="#fff"))))
    fig_ct.update_layout(title="Cost vs Time (bubble size = RL reward)", xaxis_title="Delivery time (min)",
                         yaxis_title="Cost ($)", template="plotly_white", height=340, margin=dict(l=10, r=10, t=50, b=10))
    v1.plotly_chart(fig_ct, width="stretch")

    fig_rw = go.Figure()
    fig_rw.add_trace(go.Bar(x=names, y=[r["q_value"] for r in rows], name="RL value (learned)", marker_color=colors))
    fig_rw.add_trace(go.Bar(x=names, y=[r["formula_reward"] for r in rows], name="Formula estimate",
                            marker_color="#19C3A6", opacity=.7))
    fig_rw.update_layout(title="Reward comparison", barmode="group", template="plotly_white", height=340,
                         margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
    v2.plotly_chart(fig_rw, width="stretch")

    fig_rk = go.Figure(go.Bar(x=names, y=[r["risk_score"] for r in rows],
                              marker_color=["#E4572E" if r["risk_level"] == "High" else "#F5A623" if r["risk_level"] == "Medium"
                                            else "#19C3A6" for r in rows],
                              text=[r["risk_level"] for r in rows], textposition="outside"))
    fig_rk.update_layout(title="Risk comparison (lower is better)", yaxis_title="Risk score", template="plotly_white",
                         height=340, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_rk, width="stretch")

    with st.expander("Model quality (hold-out data)"):
        mm = result["model_metrics"]
        if mm:
            st.write(f"Cost model R²: **{mm.get('cost_r2', float('nan')):.3f}** · Time model R²: "
                     f"**{mm.get('time_r2', float('nan')):.3f}** · Success classifier accuracy: "
                     f"**{mm.get('success_accuracy', float('nan')):.3f}**")
        else:
            st.write("Metrics unavailable for this dataset.")

    # ---------------------------------------------------------------- chat
    section("💬 Delivery assistant")
    st.caption("Ask about this recommendation. Answers are written by the Groq LLM from the RL engine's numbers.")

    def ask(text: str) -> None:
        st.session_state["pending_q"] = text

    quick = ["Why did you choose this option?", "What if I increase my budget?",
             "What if delivery must arrive within 2 hours?", "Which option is cheapest?",
             "Which option gives the highest reward?"]
    qcols = st.columns(len(quick))
    for col, text in zip(qcols, quick):
        col.button(text, key="q_" + text, on_click=ask, args=(text,))

    history = st.session_state.setdefault("chat", [])
    question = st.chat_input("Ask the delivery assistant...") or st.session_state.pop("pending_q", None)
    if question:
        history.append({"role": "user", "content": question})
        try:
            if not has_api_key():
                raise RuntimeError("no key")
            with st.spinner("Thinking..."):
                answer = chat_answer(question, result, st.session_state["scenarios"], history[:-1])
            if not answer:
                raise RuntimeError("empty answer")
        except Exception as exc:
            prefix = ("*(Offline answer - " + friendly_error(exc) + ")*\n\n") if has_api_key() else \
                     "*(Offline answer - add GROQ_API_KEY for AI-written replies.)*\n\n"
            answer = prefix + rule_based_answer(question, result, st.session_state["scenarios"])
        history.append({"role": "assistant", "content": answer})
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
else:
    st.info("Fill in the delivery request above and press **Optimize Delivery**.")
