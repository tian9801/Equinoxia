from forecast import (
    make_features,
    time_split,
    train_mean_models,
    train_quantile_models,
    evaluate_mean_model,
    choose_best_by_val,
    approx_dist_from_quantiles,
)

from sklearn.metrics import mean_absolute_error, mean_squared_error


import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from openai import OpenAI
from openai import RateLimitError, APIError

load_dotenv()
st.set_page_config(page_title="Equinoxia", layout="wide")

mpl.rcParams.update({
    "figure.figsize": (6.6, 3.6),
    "figure.dpi": 160,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": (1, 1, 1, 0.18),
    "axes.labelcolor": "#E6EAF2",
    "text.color": "#E6EAF2",
    "xtick.color": "#E6EAF2",
    "ytick.color": "#E6EAF2",
    "grid.color": (1, 1, 1, 0.12),
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "font.size": 10,
})


def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
@import url("https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0");

/* Space Grotesk everywhere EXCEPT icon spans */
.stApp *:not([data-testid="stIconMaterial"]) {
  font-family: "Space Grotesk", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Inter, Arial, sans-serif !important;
}

/* Kill the raw ligature text in sidebar toggle icons */
span[data-testid="stIconMaterial"] {
  font-size: 0 !important;
}

:root {
  --text: #E6EAF2;
  --muted: rgba(230,234,242,0.70);
  --blue: #3B82F6;
  --blue2: #1D4ED8;
  --purple: #7C3AED;
  --bg1: #070A14;
  --bg2: #0B1020;
  --glass: rgba(255,255,255,0.06);
  --glassBorder: rgba(255,255,255,0.16);
  --sidebarTop: rgba(17,26,51,0.90);
  --sidebarBot: rgba(11,16,32,0.92);
}

@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@keyframes glowPulse { 0% { box-shadow: 0 0 0 rgba(59,130,246,0.0); } 50% { box-shadow: 0 0 24px rgba(59,130,246,0.18); } 100% { box-shadow: 0 0 0 rgba(59,130,246,0.0); } }
@keyframes borderShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

.stApp {
  background:
    radial-gradient(1200px circle at 10% 10%, rgba(124,58,237,0.35), transparent 40%),
    radial-gradient(900px circle at 90% 20%, rgba(59,130,246,0.30), transparent 35%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 60%, var(--bg1) 100%);
  animation: fadeUp 420ms ease-out both;
}

.stApp :where(h1,h2,h3,h4,h5,h6,p,label,small,span,div) { color: var(--text); }
section[data-testid="stSidebar"] :where(h1,h2,h3,h4,h5,h6,p,label,small,span,div) { color: var(--text); }

header[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
header [data-testid="stToolbar"] { display: flex !important; background: transparent !important; box-shadow: none !important; }
header button[title="Deploy"],
header [data-testid="stDeployButton"],
header [data-testid="stStatusWidget"],
header [data-testid="stDecoration"] { display: none !important; }

section[data-testid="stSidebar"] {
  background:
    radial-gradient(900px circle at 20% 10%, rgba(124,58,237,0.18), transparent 40%),
    radial-gradient(700px circle at 90% 30%, rgba(59,130,246,0.16), transparent 45%),
    linear-gradient(180deg, var(--sidebarTop) 0%, var(--sidebarBot) 100%) !important;
  border-right: 1px solid rgba(59,130,246,0.22) !important;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: 14px 0 50px rgba(0,0,0,0.35);
  animation: fadeUp 420ms ease-out both;
}

section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stDateInput,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stMultiSelect {
  padding: 10px 12px;
  border-radius: 16px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 32px rgba(0,0,0,0.22);
  margin-bottom: 10px;
  transition: box-shadow 160ms ease, border-color 160ms ease, transform 160ms ease;
}
section[data-testid="stSidebar"] .stTextInput:hover,
section[data-testid="stSidebar"] .stNumberInput:hover,
section[data-testid="stSidebar"] .stDateInput:hover,
section[data-testid="stSidebar"] .stSlider:hover,
section[data-testid="stSidebar"] .stMultiSelect:hover {
  border-color: rgba(59,130,246,0.28);
  box-shadow: 0 14px 44px rgba(0,0,0,0.30), 0 0 18px rgba(59,130,246,0.12), 0 0 18px rgba(124,58,237,0.10);
  transform: translateY(-1px);
}

.stTextInput input, .stNumberInput input, .stDateInput input, textarea {
  background: #FFFFFF !important;
  color: #111827 !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 12px !important;
  caret-color: #111827 !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus, textarea:focus {
  outline: none !important;
  border-color: rgba(59,130,246,0.85) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.18) !important;
}

.stButton > button, button[kind="primary"], button[kind="secondary"] {
  background: linear-gradient(90deg, rgba(124,58,237,0.95), rgba(59,130,246,0.95)) !important;
  border: 1px solid rgba(59,130,246,0.55) !important;
  border-radius: 18px !important;
  color: var(--text) !important;
  font-weight: 700 !important;
  transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
  box-shadow: 0 10px 28px rgba(0,0,0,0.28);
}
.stButton > button * { color: var(--text) !important; }
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 16px 40px rgba(0,0,0,0.35), 0 0 22px rgba(124,58,237,0.25), 0 0 18px rgba(59,130,246,0.20);
  filter: saturate(1.08);
}
.stButton > button:active { transform: translateY(0px) scale(0.99); }

div[data-testid="stSlider"] {
  background: rgba(17,26,51,0.55);
  border-radius: 14px;
  border: 1px solid rgba(59,130,246,0.25);
  padding: 10px;
  transition: box-shadow 160ms ease, border-color 160ms ease;
}
div[data-testid="stSlider"]:hover { border-color: rgba(59,130,246,0.50) !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.10); }
div[data-testid="stSlider"] div[aria-hidden="true"] { background: rgba(255,255,255,0.22) !important; border-radius: 999px !important; }
div[data-testid="stSlider"] div[aria-hidden="true"] div { background: linear-gradient(90deg, var(--purple), var(--blue)) !important; border-radius: 999px !important; }
div[data-testid="stSlider"] [role="slider"] { background: #FFFFFF !important; border: 2px solid rgba(59,130,246,0.95) !important; }

div[data-baseweb="select"] > div { background: var(--glass) !important; border: 1px solid var(--glassBorder) !important; border-radius: 14px !important; }
div[data-baseweb="select"] input { color: var(--text) !important; }
div[data-baseweb="select"] svg { fill: rgba(230,234,242,0.85) !important; }

div[data-baseweb="tag"] { background: rgba(124,58,237,0.22) !important; border: 1px solid rgba(59,130,246,0.65) !important; border-radius: 999px !important; }
div[data-baseweb="tag"] span { color: var(--text) !important; font-weight: 700 !important; }
div[data-baseweb="tag"] svg { fill: var(--text) !important; opacity: 0.95 !important; }

div[data-testid="stMultiSelectVirtualDropdown"] {
  background: rgba(11,16,32,0.92) !important;
  border: 1px solid rgba(59,130,246,0.40) !important;
  border-radius: 14px !important;
  box-shadow: 0 20px 70px rgba(0,0,0,0.50) !important;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  animation: fadeUp 220ms ease-out both;
}
div[data-testid="stMultiSelectVirtualDropdown"] * { color: var(--text) !important; opacity: 1 !important; }
div[data-testid="stMultiSelectVirtualDropdown"] li { background: transparent !important; border-radius: 10px !important; transition: background 140ms ease, transform 140ms ease; }
div[data-testid="stMultiSelectVirtualDropdown"] li:hover { background: rgba(59,130,246,0.18) !important; transform: translateX(2px); }

.card {
  position: relative;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 18px 18px;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow: 0 14px 50px rgba(0,0,0,0.35);
  animation: fadeUp 420ms ease-out both;
  overflow: hidden;
}
.card::before {
  content: "";
  position: absolute;
  inset: -1px;
  border-radius: 18px;
  padding: 1px;
  background: linear-gradient(90deg, rgba(124,58,237,0.65), rgba(59,130,246,0.65), rgba(124,58,237,0.65));
  background-size: 200% 200%;
  animation: borderShift 6s ease infinite;
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
  opacity: 0.85;
}
.card:hover { box-shadow: 0 18px 60px rgba(0,0,0,0.42), 0 0 26px rgba(59,130,246,0.16), 0 0 26px rgba(124,58,237,0.14); }

button[aria-label="View fullscreen"] svg, button[aria-label="Fullscreen"] svg { display: none !important; }
button[aria-label="View fullscreen"], button[aria-label="Fullscreen"] {
  background: rgba(17,26,51,0.90) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 12px !important;
  transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
}
button[aria-label="View fullscreen"]::after, button[aria-label="Fullscreen"]::after {
  content: "⤢";
  color: var(--text);
  font-size: 18px;
  display: inline-block;
  transform: translateY(-1px);
  text-shadow: 0 0 10px rgba(59,130,246,0.28), 0 0 10px rgba(124,58,237,0.22);
}
button[aria-label="View fullscreen"]:hover, button[aria-label="Fullscreen"]:hover {
  transform: translateY(-1px) scale(1.03);
  border-color: rgba(59,130,246,0.55) !important;
  box-shadow: 0 10px 26px rgba(0,0,0,0.35), 0 0 18px rgba(59,130,246,0.22), 0 0 18px rgba(124,58,237,0.18);
  animation: glowPulse 1.6s ease-in-out infinite;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ===== HAMBURGER — both open and collapsed states ===== */

/* wrapper divs: never hide either of them */
div[data-testid="stSidebarCollapseButton"],
div[data-testid="stExpandSidebarButton"] {
  opacity: 1 !important;
  visibility: visible !important;
  display: flex !important;
  pointer-events: all !important;
}

/* shared button style */
div[data-testid="stSidebarCollapseButton"] button,
div[data-testid="stExpandSidebarButton"] button {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  width: 40px !important;
  height: 40px !important;
  min-width: 40px !important;
  min-height: 40px !important;
  padding: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  overflow: visible !important;
  font-size: 0 !important;
  color: transparent !important;
  opacity: 1 !important;
  visibility: visible !important;
}

/* hide streamlit's icon span inside both buttons */
div[data-testid="stSidebarCollapseButton"] button *,
div[data-testid="stExpandSidebarButton"] button * {
  display: none !important;
}

/* hamburger icon on both via ::after */
div[data-testid="stSidebarCollapseButton"] button::after,
div[data-testid="stExpandSidebarButton"] button::after {
  content: "menu" !important;
  font-family: "Material Symbols Rounded" !important;
  font-variation-settings: "opsz" 24, "wght" 700, "FILL" 0, "GRAD" 0 !important;
  -webkit-font-feature-settings: "liga" !important;
  font-feature-settings: "liga" !important;
  font-size: 26px !important;
  line-height: 1 !important;
  color: #E6EAF2 !important;
  display: block !important;
  opacity: 1 !important;
  visibility: visible !important;
}



</style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


@st.cache_data(show_spinner=False)
def fetch_prices(tickers, start="2018-01-01"):
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


def make_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def estimate_betas(port_ret: pd.Series, factor_rets: pd.DataFrame):
    X = factor_rets.values
    y = port_ret.values
    model = LinearRegression().fit(X, y)
    beta = pd.Series(model.coef_, index=factor_rets.columns)
    alpha = float(model.intercept_)
    resid = y - model.predict(X)
    resid_std = float(np.std(resid, ddof=1))
    return alpha, beta, resid_std


def simulate_portfolio(alpha, beta: pd.Series, resid_std, factor_mean, factor_cov, n_sims=5000, scenario_shift=None):
    if scenario_shift is None:
        scenario_shift = pd.Series(0.0, index=beta.index)
    mu = factor_mean + scenario_shift
    sims_f = np.random.multivariate_normal(mean=mu.values, cov=factor_cov.values, size=n_sims)
    sims_eps = np.random.normal(loc=0.0, scale=resid_std, size=n_sims)
    sims_r = alpha + sims_f @ beta.values + sims_eps
    return sims_r


def var_cvar(returns, level=0.95):
    q = np.quantile(returns, 1 - level)
    tail = returns[returns <= q]
    cvar = tail.mean() if len(tail) else q
    return float(q), float(cvar)


def safe_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None, "OPENAI_API_KEY not set."
    return OpenAI(api_key=key), None


def openai_parse_scenario(user_text, factor_names):
    client, err = safe_openai_client()
    if err:
        return None, err

    prompt = f"""
Convert a user's market scenario into numeric DAILY return shocks for these factors:
{factor_names}

Rules:
- Values must be decimal returns (e.g., -0.02 for -2%).
- If "VIX up X%", treat it as +X% return on the VIX factor (approx).
- User gives no value for a factor, set it to 0.

User scenario: {user_text}
""".strip()

    try:
        resp = client.responses.create(model="gpt-5", input=prompt)
        obj = json.loads(resp.output_text.strip())
        for k in factor_names:
            obj.setdefault(k, 0.0)
        return pd.Series(obj).reindex(factor_names).astype(float), None
    except RateLimitError:
        return None, "OpenAI quota/billing not available for this API key (429 insufficient_quota)."
    except APIError as e:
        return None, f"OpenAI API error: {e}"
    except Exception as e:
        return None, f"Could not parse JSON from model output: {e}"


# SIDEBAR PROGRAMMING
st.markdown("# Equinoxia")
st.caption("Stress test your portfolio with factor shocks + Monte Carlo simulation.")

st.sidebar.header("Portfolio")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "AAPL,MSFT,NVDA")
weights_input = st.sidebar.text_input("Weights (comma separated, sum≈1)", "0.34,0.33,0.33")
start_date = st.sidebar.text_input("Start date (YYYY-MM-DD)", "2018-01-01")

st.sidebar.header("Factors")
factor_choices = {
    "SPY (market)": "SPY",
    "^VIX (vol)": "^VIX",
    "IEF (rates proxy)": "IEF",
    "USO (oil proxy)": "USO",
    "UUP (USD proxy)": "UUP",
}
factor_selected = st.sidebar.multiselect(
    "Choose factor tickers",
    list(factor_choices.values()),
    default=["SPY", "^VIX", "IEF", "USO"],
)

st.sidebar.header("Simulation")
n_sims = st.sidebar.slider("Simulations", 1000, 20000, 8000, 1000)
level = st.sidebar.slider("VaR confidence", 0.80, 0.99, 0.95, 0.01)

st.sidebar.header("Stock Forecast")
lookback = st.sidebar.slider("Feature lookback (days)", 5, 60, 20, 5)
train_frac = st.sidebar.slider("Train fraction", 0.50, 0.85, 0.70, 0.05)
val_frac = st.sidebar.slider("Val fraction", 0.05, 0.30, 0.15, 0.05)
use_quantiles = st.sidebar.checkbox("Predict quantiles (VaR)", value=True)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
weights = np.array([float(x.strip()) for x in weights_input.split(",") if x.strip()], dtype=float)

if len(tickers) == 0:
    st.error("Enter at least one ticker.")
    st.stop()
if len(weights) != len(tickers):
    st.error("Weights count must match tickers count.")
    st.stop()
weights = weights / weights.sum()
if len(factor_selected) == 0:
    st.error("Select at least one factor ticker.")
    st.stop()



all_tickers = list(dict.fromkeys(tickers + factor_selected))
with st.spinner("Loading price data..."):
    prices = fetch_prices(all_tickers, start=start_date)

rets = make_returns(prices)

missing_assets = [t for t in tickers if t not in rets.columns]
missing_factors = [f for f in factor_selected if f not in rets.columns]
if missing_assets:
    st.error(f"Missing asset data for: {missing_assets}")
    st.stop()
if missing_factors:
    st.error(f"Missing factor data for: {missing_factors}")
    st.stop()

asset_rets = rets[tickers].dropna()
factor_rets = rets[factor_selected].dropna()
common_index = asset_rets.index.intersection(factor_rets.index)
asset_rets = asset_rets.loc[common_index]
factor_rets = factor_rets.loc[common_index]
port_ret = (asset_rets * weights).sum(axis=1).dropna()

alpha, beta, resid_std = estimate_betas(port_ret, factor_rets)

#SCENARIO STATE CODE
if "scenario_shift" not in st.session_state:
    st.session_state.scenario_shift = {}
for f in factor_selected:
    st.session_state.scenario_shift.setdefault(f, 0.0)
for k in list(st.session_state.scenario_shift.keys()):
    if k not in factor_selected:
        del st.session_state.scenario_shift[k]


# -TABS
tab_overview, tab_scenario, tab_results, tab_model, tab_data, tab_forecast, tab_explain = st.tabs(
    ["Overview", "Scenario", "Results", "Exposures", "Data", "Forecast", "Explain"]
)

with tab_overview:
    st.subheader("What this does")
    st.write(
        "We fit a linear factor model to your portfolio's daily returns, then simulate outcomes under a scenario "
        "by sampling factor moves and residual noise. Output is a distribution, not a single prediction."
    )
    st.write("**Tickers:**", ", ".join(tickers))
    st.write("**Factors:**", ", ".join(factor_selected))

with tab_model:
    st.subheader("Factor exposures")
    st.write("Alpha (daily):", alpha)
    st.write("Betas:")
    st.dataframe(beta.to_frame("beta"))
    st.caption("Interpretation: SPY beta dominates = portfolio moves strongly with the market.")

with tab_data:
    st.subheader("Data")
    st.write("Price data (tail):")
    st.dataframe(prices.tail())

with tab_scenario:
    st.subheader("Scenario builder")

    mode = st.radio(
        "Input mode",
        ["Manual (percent sliders)", "Describe in English (OpenAI)"],
        horizontal=True,
    )

    st.caption("Use percent shocks like -2.0 meaning -2% daily return shock for that factor.")

    templates = {
        "Reset": {f: 0.0 for f in factor_selected},
        "Risk-off": {"SPY": -2.5, "^VIX": 15.0, "IEF": 0.6, "USO": -1.5, "UUP": 0.4},
        "Inflation scare": {"SPY": -2.0, "^VIX": 10.0, "IEF": -0.8, "USO": 6.0, "UUP": 0.5},
        "Rate cut rally": {"SPY": 1.5, "^VIX": -8.0, "IEF": 0.9, "USO": 0.0, "UUP": -0.3},
        "Oil shock": {"SPY": -1.0, "^VIX": 6.0, "IEF": -0.2, "USO": 10.0, "UUP": 0.2},
    }

    tcols = st.columns(5)
    tnames = list(templates.keys())[:5]
    for i, name in enumerate(tnames):
        if tcols[i].button(name):
            for f in factor_selected:
                st.session_state.scenario_shift[f] = float(templates[name].get(f, 0.0))

    if mode == "Manual (percent sliders)":
        cols = st.columns(len(factor_selected))
        for i, f in enumerate(factor_selected):
            with cols[i]:
                st.session_state.scenario_shift[f] = st.number_input(
                    f"{f} (%)",
                    value=float(st.session_state.scenario_shift.get(f, 0.0)),
                    step=0.5,
                    format="%.2f",
                )
    else:
        user_text = st.text_area(
            "Describe scenario",
            value="SPY down 2%, VIX up 15%, oil up 8%",
            height=90,
        )
        if st.button("Parse scenario"):
            parsed, err = openai_parse_scenario(user_text, list(factor_selected))
            if err:
                st.error(err)
            else:
                for f in factor_selected:
                    st.session_state.scenario_shift[f] = float(parsed[f] * 100.0)
                st.success("Parsed shocks applied.")
                st.json({k: float(v) for k, v in st.session_state.scenario_shift.items()})

with tab_results:
    st.subheader("Results")

    scenario_shift = pd.Series({f: st.session_state.scenario_shift[f] / 100.0 for f in factor_selected})
    factor_mean = factor_rets.mean()
    factor_cov = factor_rets.cov()

    sims = simulate_portfolio(
        alpha, beta, resid_std,
        factor_mean, factor_cov,
        n_sims=n_sims,
        scenario_shift=scenario_shift,
    )

    v, c = var_cvar(sims, level=level)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean return", f"{np.mean(sims)*100:.2f}%")
    c2.metric(f"VaR {int(level*100)}%", f"{v*100:.2f}%")
    c3.metric(f"CVaR {int(level*100)}%", f"{c*100:.2f}%")
    c4.metric("P(return < -1%)", f"{np.mean(sims < -0.01)*100:.1f}%")

    contrib = (beta * scenario_shift).sort_values()
    fig_c, ax = plt.subplots(figsize=(6.6, 3.2), dpi=160)
    ax.barh(contrib.index, contrib.values)
    ax.set_title("Factor contribution (beta × shock)", pad=10)
    ax.set_xlabel("Return contribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_c.tight_layout()
    st.pyplot(fig_c, clear_figure=True, width="content")

    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=160)
    ax.hist(sims, bins=45)
    ax.set_title("Simulated daily portfolio returns", pad=10)
    ax.set_xlabel("Return")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="content")



with tab_forecast:
    st.subheader("Forecast: next-day return")
    st.caption("Train models on rolling factor features to predict the portfolio's next-day return.")

    X, y = make_features(factor_rets, port_ret, lookback=lookback)
    #just in case
    if len(X) < 200:
        st.warning("Not enough data after feature building, try again with an earlier start date or fewer windows.")
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(
            X, y, train_frac=train_frac, val_frac=val_frac
        )
        models = train_mean_models(X_train, y_train, random_state=0)

        rows = []
        for name, m in models.items():
            val_metrics = evaluate_mean_model(m, X_val, y_val)
            test_metrics = evaluate_mean_model(m, X_test, y_test)
            rows.append({
                "model": name,
                "val_MAE": val_metrics["MAE"],
                "val_RMSE": val_metrics["RMSE"],
                "test_MAE": test_metrics["MAE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_direction_acc": test_metrics["DirectionAcc"],
            })

        st.dataframe(pd.DataFrame(rows).set_index("model"))

        #picking best by validation RMSE
        best_name, best_model, best_val_rmse = choose_best_by_val(models, X_val, y_val)
        st.caption(f"Using **{best_name}** (lowest val RMSE = {best_val_rmse:.6f}) for live forecast.")

        #live prediction (next day) w/ most recent feature row
        x_last = X.iloc[[-1]]
        mean_pred = float(best_model.predict(x_last)[0])

        c1, c2 = st.columns(2)
        c1.metric("Predicted next-day mean return", f"{mean_pred*100:.2f}%")
        c2.metric("Last observed portfolio return", f"{float(port_ret.iloc[-1])*100:.2f}%")

        #quantile prediction for VaR bounds
        if use_quantiles:
            q_models = train_quantile_models(X_train, y_train, qs=(0.05, 0.50, 0.95))
            q05 = float(q_models["Q5"].predict(x_last)[0])
            q50 = float(q_models["Q50"].predict(x_last)[0])
            q95 = float(q_models["Q95"].predict(x_last)[0])

            q1, q2, q3 = st.columns(3)
            q1.metric("Q5 (downside)", f"{q05*100:.2f}%")
            q2.metric("Q50 (median)", f"{q50*100:.2f}%")
            q3.metric("Q95 (upside)", f"{q95*100:.2f}%")

            # Make an approximate distribution and reuse your VaR/CVaR code
            sims_ml = approx_dist_from_quantiles(q05, q50, q95, n=n_sims, seed=0)
            v_ml, c_ml = var_cvar(sims_ml, level=level)

            d1, d2, d3 = st.columns(3)
            d1.metric(f"Implied VaR {int(level*100)}%", f"{v_ml*100:.2f}%")
            d2.metric(f"Implied CVaR {int(level*100)}%", f"{c_ml*100:.2f}%")
            d3.metric("P(return < -1%)", f"{np.mean(sims_ml < -0.01)*100:.1f}%")

            fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=160)
            ax.hist(sims_ml, bins=45)
            ax.set_title("ML-implied next-day return distribution", pad=10)
            ax.set_xlabel("Return")
            ax.set_ylabel("Count")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True, width="content")

with tab_explain:
    st.subheader("Explain (OpenAI)")
    st.caption("Plain English explanation of the scenario + outputs.")

    if st.button("Explain results"):
        client, err = safe_openai_client()
        if err:
            st.error(err)
        else:
            scenario_shift_out = {f: float(st.session_state.scenario_shift[f] / 100.0) for f in factor_selected}
            summary = {
                "tickers": tickers,
                "weights": [float(w) for w in weights.tolist()],
                "factors": list(factor_selected),
                "betas": {k: float(v) for k, v in beta.items()},
                "scenario_shift_decimal_returns": scenario_shift_out,
                "alpha_daily": float(alpha),
                "var": float(v),
                "cvar": float(c),
                "mean_sim_return": float(np.mean(sims)),
                "prob_loss_gt_1pct": float(np.mean(sims < -0.01)),
            }

            try:
                resp = client.responses.create(
                    model="gpt-5",
                    input=(
                        "Explain this scenario risk result in clear English. "
                        "Reference the JSON values and include caveats that this is a simple factor model.\n\n"
                        + json.dumps(summary, indent=2)
                    ),
                )
                st.write(resp.output_text)
            except RateLimitError:
                st.error("OpenAI quota/billing not available for this API key (429 insufficient_quota).")
            except APIError as e:
                st.error(f"OpenAI API error: {e}")