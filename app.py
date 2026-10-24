import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Verdikt Pro | Polymarket UFC Terminal", page_icon="📈", layout="wide")

DEFAULT_ROSTER = {
    "Renato Moicano": {"elo": 1745, "age": 37.4, "reach": 72.0, "dmg_ratio": 1.24, "slpm": 4.40, "td": 1.80, "cardio": 1.2, "layoff": 210, "cardio_nan": False, "short_notice": False},
    "Tom Nolan": {"elo": 1640, "age": 26.6, "reach": 73.0, "dmg_ratio": 1.25, "slpm": 4.68, "td": 1.04, "cardio": 0.8, "layoff": 168, "cardio_nan": False, "short_notice": False},
    "Randy Brown": {"elo": 1670, "age": 36.3, "reach": 78.0, "dmg_ratio": 1.35, "slpm": 4.52, "td": 0.73, "cardio": 0.5, "layoff": 154, "cardio_nan": False, "short_notice": False},
    "Jacobe Smith": {"elo": 1615, "age": 30.8, "reach": 72.0, "dmg_ratio": 2.38, "slpm": 6.45, "td": 4.88, "cardio": 0.0, "layoff": 28, "cardio_nan": True, "short_notice": True},
    "Alexander Volkanovski": {"elo": 1825, "age": 38.0, "reach": 71.5, "dmg_ratio": 1.81, "slpm": 6.12, "td": 1.65, "cardio": -1.5, "layoff": 210, "cardio_nan": False, "short_notice": False},
    "Movsar Evloev": {"elo": 1735, "age": 32.0, "reach": 72.5, "dmg_ratio": 1.38, "slpm": 3.95, "td": 4.65, "cardio": -4.2, "layoff": 240, "cardio_nan": False, "short_notice": False},
    "Custom Fighter": {"elo": 1500, "age": 29.0, "reach": 70.0, "dmg_ratio": 1.15, "slpm": 3.80, "td": 1.50, "cardio": 0.0, "layoff": 150, "cardio_nan": False, "short_notice": False}
}

st.sidebar.title("📈 Polymarket Portfolio Controls")
bankroll = st.sidebar.number_input("USDC Bankroll ($)", min_value=10.0, value=1000.0, step=50.0)
kelly_mult = st.sidebar.selectbox("Kelly Multiplier", [0.25, 0.50, 0.125], index=0, format_func=lambda x: f"{x}x (Quarter-Kelly)" if x==0.25 else f"{x}x")
min_edge = st.sidebar.slider("Min Absolute Probability Edge (%)", 1.0, 15.0, 4.0, 0.5) / 100.0
max_bet_cap = st.sidebar.slider("Max Single-Market Exposure (%)", 1.0, 15.0, 5.0, 0.5) / 100.0

def predict_symmetric_probability(f1: dict, f2: dict):
    d_fwd = {
        "Delta_Elo": f1["elo"] - f2["elo"], "Delta_Age": f1["age"] - f2["age"],
        "Delta_Reach": f1["reach"] - f2["reach"], "Delta_Damage_Ratio": f1["dmg_ratio"] - f2["dmg_ratio"],
        "Delta_Weighted_SLpM": f1["slpm"] - f2["slpm"], "Delta_Weighted_TD": f1["td"] - f2["td"],
        "Delta_Cardio_Deg": 0.0 if (f1["cardio_nan"] or f2["cardio_nan"]) else (f1["cardio"] - f2["cardio"]),
        "Delta_Layoff": f1["layoff"] - f2["layoff"], "Chaos_Adjustment": f1["chaos_mod"] - f2["chaos_mod"]
    }
    d_rev = {k: -v for k, v in d_fwd.items()}
    logit_fwd = 0.0055*d_fwd["Delta_Elo"] - 0.052*d_fwd["Delta_Age"] + 0.018*d_fwd["Delta_Reach"] + 0.62*d_fwd["Delta_Damage_Ratio"] + 0.085*d_fwd["Delta_Weighted_SLpM"] + 0.065*d_fwd["Delta_Weighted_TD"] + 0.045*d_fwd["Delta_Cardio_Deg"] - 0.0006*d_fwd["Delta_Layoff"] + d_fwd["Chaos_Adjustment"]
    logit_rev = 0.0055*d_rev["Delta_Elo"] - 0.052*d_rev["Delta_Age"] + 0.018*d_rev["Delta_Reach"] + 0.62*d_rev["Delta_Damage_Ratio"] + 0.085*d_rev["Delta_Weighted_SLpM"] + 0.065*d_rev["Delta_Weighted_TD"] + 0.045*d_rev["Delta_Cardio_Deg"] - 0.0006*d_rev["Delta_Layoff"] + d_rev["Chaos_Adjustment"]
    p_fwd, p_rev = 1.0/(1.0+np.exp(-logit_fwd)), 1.0/(1.0+np.exp(-logit_rev))
    p_f1 = 0.5 * (p_fwd + (1.0 - p_rev))
    return p_f1, 1.0 - p_f1

def calculate_polymarket_kelly(model_prob: float, share_price_cents: float, bankroll: float, kelly_mult: float, max_cap: float):
    p_share = share_price_cents / 100.0
    edge = model_prob - p_share
    if edge <= 0 or p_share >= 1.0 or p_share <= 0:
        return {"edge": edge, "bet_pct": 0.0, "investment": 0.0, "shares": 0}
    full_kelly = edge / (1.0 - p_share)
    bet_pct = min(full_kelly * kelly_mult, max_cap)
    investment = round(bankroll * bet_pct, 2)
    shares = int(investment / p_share) if p_share > 0 else 0
    return {"edge": edge, "bet_pct": bet_pct, "investment": investment, "shares": shares}

st.title("🎯 Polymarket UFC Quant Terminal")
col1, col2 = st.columns(2)

def render_fighter(col, label, default_idx):
    with col:
        st.subheader(label)
        choice = st.selectbox(f"Select {label}", list(DEFAULT_ROSTER.keys()), index=default_idx, key=f"sel_{label}")
        preset = DEFAULT_ROSTER[choice]
        name = st.text_input("Fighter Name", value=choice, key=f"name_{label}")
        share_price = st.number_input("Polymarket YES Price (1¢ to 99¢)", min_value=1.0, max_value=99.0, value=50.0, step=1.0, key=f"price_{label}")
        with st.expander("Fighter Metrics", expanded=False):
            elo = st.number_input("Elo", 1000.0, 2200.0, float(preset["elo"]), key=f"elo_{label}")
            age = st.number_input("Age", 18.0, 50.0, float(preset["age"]), key=f"age_{label}")
            reach = st.number_input("Reach", 55.0, 88.0, float(preset["reach"]), key=f"reach_{label}")
            dmg_ratio = st.number_input("Damage Ratio", 0.1, 5.0, float(preset["dmg_ratio"]), key=f"dmg_{label}")
            slpm = st.number_input("SLpM", 0.5, 12.0, float(preset["slpm"]), key=f"slpm_{label}")
            td = st.number_input("TD / 15m", 0.0, 12.0, float(preset["td"]), key=f"td_{label}")
            cardio_nan = st.checkbox("Untested R3 Cardio", value=preset["cardio_nan"], key=f"cnan_{label}")
            cardio = st.number_input("Cardio Delta", -20.0, 20.0, float(preset["cardio"]), disabled=cardio_nan, key=f"card_{label}")
            layoff = st.number_input("Layoff (Days)", 7.0, 1500.0, float(preset["layoff"]), key=f"lay_{label}")
        sn = st.checkbox("Short-Notice (<14 Days)", value=preset["short_notice"], key=f"sn_{label}")
        bs = st.checkbox("Missed Weight / Drain", value=False, key=f"bs_{label}")
        chaos_mod = (-0.28 if sn else 0.0) + (-0.22 if bs else 0.0)
        return {"name": name, "share_price": share_price, "elo": elo, "age": age, "reach": reach, "dmg_ratio": dmg_ratio, "slpm": slpm, "td": td, "cardio": cardio, "cardio_nan": cardio_nan, "layoff": layoff, "chaos_mod": chaos_mod}

f1 = render_fighter(col1, "Fighter A (Red Corner)", 0)
f2 = render_fighter(col2, "Fighter B (Blue Corner)", 1)
st.divider()

p_f1, p_f2 = predict_symmetric_probability(f1, f2)
trade_f1 = calculate_polymarket_kelly(p_f1, f1["share_price"], bankroll, kelly_mult, max_bet_cap)
trade_f2 = calculate_polymarket_kelly(p_f2, f2["share_price"], bankroll, kelly_mult, max_bet_cap)

m1, m2 = st.columns(2)
m1.metric(f"{f1['name']} (Market: {f1['share_price']:.0f}¢)", f"{p_f1*100:.1f}% Model Prob", f"{trade_f1['edge']*100:+.1f}% Edge")
m2.metric(f"{f2['name']} (Market: {f2['share_price']:.0f}¢)", f"{p_f2*100:.1f}% Model Prob", f"{trade_f2['edge']*100:+.1f}% Edge")

if trade_f1["edge"] >= min_edge and trade_f1["edge"] > trade_f2["edge"]:
    st.success(f"### 🟢 BUY YES SHARES: {f1['name'].upper()} @ {f1['share_price']:.0f}¢\n- **Model Probability:** `{p_f1*100:.1f}%`\n- **Allocation:** `USDC ${trade_f1['investment']:,.2f}` (`{trade_f1['bet_pct']*100:.1f}%` Bankroll) — **~{trade_f1['shares']} shares**")
elif trade_f2["edge"] >= min_edge and trade_f2["edge"] > trade_f1["edge"]:
    st.success(f"### 🟢 BUY YES SHARES: {f2['name'].upper()} @ {f2['share_price']:.0f}¢\n- **Model Probability:** `{p_f2*100:.1f}%`\n- **Allocation:** `USDC ${trade_f2['investment']:,.2f}` (`{trade_f2['bet_pct']*100:.1f}%` Bankroll) — **~{trade_f2['shares']} shares**")
else:
    st.warning("### ⚪ HOLD / PASS — Market is Efficient (No Edge Above Threshold)")
