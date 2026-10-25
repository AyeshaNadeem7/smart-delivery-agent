# 🚚 SmartRoute AI — RL + LLM Delivery Optimization Agent

A portfolio-ready project that recommends the **best delivery option** for a package — not just the cheapest or the fastest — by combining:

| Layer | Technology | Job |
|---|---|---|
| Decision | **Reinforcement learning** (contextual bandit, Q-learning updates) | Chooses the option with the best expected overall reward |
| Prediction | **scikit-learn** (linear regression + logistic regression) | Estimates cost, time and success probability per option |
| Orchestration | **CrewAI** (5 agents, sequential process) | Passes the analysis between specialists |
| Language | **Groq LLM** | Understands the request, explains the decision, powers the chat |
| Interface | **Streamlit** + Plotly | Light, modern dashboard |
| Data | Synthetic generator **or** your own CSV | The delivery "environment" the RL agent learns from |

> **Key principle:** the RL engine decides, the LLM explains. The LLM never picks the option and never claims to have run RL.

---

## Features

- Delivery form: weight, distance, priority, customer type, traffic, weather, package type, budget, deadline, available options, driver status
- Recommendation with cost, time, expected reward, risk level, satisfaction estimate and overall score
- "Why was this option selected?" — reasons computed by code (lowest cost, fastest acceptable, reliability, reward, risk)
- Comparison table of all options + Plotly charts (cost vs time, reward, risk, RL reward progression)
- RL panel: episodes, average reward, best reward, selected action, reward improvement vs. baseline
- Five CrewAI agents with individual reports
- Chat assistant (Groq) with what-if answers pre-computed by the RL engine
- Works with the built-in synthetic data or an uploaded CSV
- Graceful fallbacks: missing API key, bad CSV, LLM/CrewAI errors → the app still gives the RL recommendation

## Architecture

```
Streamlit UI (app.py)
   │  form + CSV/demo data
   ▼
data_generator.py / utils.validate_dataframe     ← delivery history (synthetic or CSV)
   ▼
estimator.py   (sklearn: cost, time, success probability per option)
rl_agent.py    (contextual-bandit RL: learns Q-values from historical rewards)
   │   recommend(): constraints (budget, deadline, weight) + argmax Q-value  ← THE DECISION
   ▼
crew.py + agents.py + tasks.py   (CrewAI, Groq LLM)
   Delivery Analyst → Cost & Time Analyst → Risk Analyst → RL Decision Analyst → Delivery Manager
   ▼
Final explanation, comparison, charts, chat assistant
```

```
delivery-ai-agent/
├── app.py              Streamlit UI
├── agents.py           Groq LLM connection + 5 CrewAI agents
├── tasks.py            CrewAI tasks + text briefs built from the real numbers
├── crew.py             Crew execution, chat assistant, offline fallbacks
├── rl_agent.py         RL agent + decision layer (recommend, what-if)
├── estimator.py        scikit-learn cost / time / success models
├── data_generator.py   Synthetic delivery data
├── utils.py            Constants, reward function, CSV validation
├── requirements.txt
├── sample_data/delivery_sample.csv
└── .streamlit/         config.toml (theme) + secrets.toml.example
```

## How the RL works

- **State:** priority, customer type, traffic, weather bucket, distance bucket, weight bucket.
- **Action:** a delivery option (Economy Bike, Standard Van, ...).
- **Reward:** `profit − cost_penalty − delay_penalty − risk_penalty + customer_satisfaction` (see `utils.compute_reward`; for CSV data it is your `delivery_reward` column).
- **Learning:** offline replay. Each episode draws a historical delivery, the agent picks an action (ε-greedy, ε decays 1.0 → 0.05) and updates its Q-table with `Q ← Q + α (reward − Q)` when its action matches what really happened in that row. A fine → coarse → global hierarchy shrinks estimates toward broader groups when data is sparse.
- **Evaluation:** 20 % of the data is held out; the learned greedy policy is compared with the historical policy (mean reward) → "reward improvement".
- **Decision:** among options that satisfy budget, deadline and weight limits, the one with the highest Q-value is recommended. If nothing satisfies both limits, the closest match is shown with a warning.

## How CrewAI works here

Five agents run as a **sequential** crew; each task receives previous outputs through `context=[...]`:
Delivery Analyst → Cost & Time Analyst → Risk & Traffic Analyst → RL Decision Analyst → Delivery Manager.
The numbers (estimates, risk, Q-values) are computed in Python and given to the agents as *system data*, so the LLM interprets and explains them but cannot change them. The manager is instructed to recommend exactly the RL-selected option.

## How the LLM is used

Groq's OpenAI-compatible endpoint is used through CrewAI's `LLM` class. The LLM: summarises requirements, explains cost/time and risk, narrates the RL result, writes the final recommendation and answers chat questions. It is **not** used to choose an option or to compute rewards.

---

## Installation (local)

Requires **Python 3.10 – 3.13**.

```bash
# 1. Create the project folder (or unzip the download) and enter it
cd delivery-ai-agent

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Environment variables / secrets

Get a free key at <https://console.groq.com/keys>.

| Name | Required | Default |
|---|---|---|
| `GROQ_API_KEY` | for AI explanations & chat | — |
| `GROQ_MODEL` | no | `llama-3.3-70b-versatile` |

**Option A – Streamlit secrets file (recommended):**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# open .streamlit/secrets.toml and paste your key
```
**Option B – environment variable:**
```bash
export GROQ_API_KEY="your-key"       # Windows PowerShell: $env:GROQ_API_KEY="your-key"
```
If a model is retired, pick a current one from <https://console.groq.com/docs/models> and set `GROQ_MODEL` — no code change needed.

## Run locally

```bash
streamlit run app.py
```
Open <http://localhost:8501>. Without a key the app still works (RL + offline explanations).

## Deploy: GitHub → Streamlit Community Cloud

1. Create a GitHub repo at <https://github.com/new> (e.g. `smartroute-ai`), leave it empty.
2. Upload the code:
   ```bash
   git init
   git add .
   git commit -m "SmartRoute AI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/smartroute-ai.git
   git push -u origin main
   ```
   (Or use *Add file → Upload files* on GitHub and drag in the folder contents.) `.gitignore` keeps `secrets.toml` out of the repo.
3. Go to <https://share.streamlit.io>, sign in with GitHub → **Create app** → *Deploy a public app from GitHub*.
4. Choose your repo, branch `main`, main file `app.py`.
5. Open **Advanced settings**: choose **Python 3.12** and paste in *Secrets*:
   ```toml
   GROQ_API_KEY = "your-key"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```
6. Click **Deploy**. First build takes a few minutes (CrewAI is a large package).

## Example usage / test scenarios

| Scenario | Inputs | What to look for |
|---|---|---|
| Everyday parcel | 3 kg, 12 km, Standard, Medium traffic, Clear, $30, 240 min | A cheap, reliable option |
| Urgent long trip | 3 kg, 35 km, Urgent, $80, 120 min | A fast option; slow ones excluded |
| Heavy package | 60 kg, 20 km, $100, 600 min | Bike / courier excluded by weight limit |
| Storm day | 2 kg, 6 km, Storm | Weather-sensitive options score lower |
| Impossible limits | $3 budget, 20 min | "Closest match" warning |
| Chat | "What if I increase my budget?" · "What if delivery must arrive within 2 hours?" | Answers from computed what-if scenarios |

CSV format: see `sample_data/delivery_sample.csv`. Required columns: `distance_km, package_weight, delivery_priority, customer_type, traffic_level, weather_risk, driver_available, driver_rating, delivery_option, estimated_time, delivery_cost, delivery_reward, delivery_success`.

## Troubleshooting

- **Build fails on Streamlit Cloud:** set Python to 3.12 in Advanced settings and redeploy.
- **Groq rate-limit (429):** free-tier limit — wait a minute or set `GROQ_MODEL=llama-3.1-8b-instant`.
- **"Model not found":** update `GROQ_MODEL` to a model listed on the Groq models page.

## Future improvements

- Real traffic / weather APIs (free tiers) instead of user-selected values
- Online learning: log real outcomes and keep updating Q-values
- Deep or linear function approximation (e.g. LinearUCB) for continuous states
- Multi-package batching and driver assignment (routing)
- Saving history to SQLite/CSV, authentication, A/B testing of policies
- CrewAI tools so agents can query the estimator directly
