import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Verdikt Pro | Polymarket UFC Terminal", page_icon="📈", layout="wide")

# =========================================================================
# SMART CSV PARSER (Extracts fighters from the fight-by-fight dataset)
# =========================================================================
@st.cache_data
def load_fighter_database():
    try:
        df_fights = pd.read_csv("ufc_fighters_cleaned.csv")
        
        # Extract unique fighters from f_1_name and f_2_name
        fighters = pd.concat([df_fights['f_1_name'], df_fights['f_2_name']]).dropna().unique()
        fighters = sorted([str(f).strip() for f in fighters if str(f).strip() != ""])
        
        # Build a robust default lookup dictionary for them
        roster = {}
        for name in fighters:
            roster[name] = {
                "elo": 1700.0, "age": 29.0, "reach": 70.0, "dmg_ratio": 1.30,
                "slpm": 4.50, "td": 1.50, "cardio": 0.0, "layoff": 150.0,
                "cardio_nan": False, "short_notice": False
            }
        return roster
    except Exception as e:
        # Fallback dictionary if file has any issue
        return {
            "Joshua Van": {"elo": 1735, "age": 24.5, "reach": 67.0, "dmg_ratio": 1.48, "slpm": 7.40, "td": 0.50, "cardio": 3.2, "layoff": 110, "cardio_nan": False, "short_notice": False},
            "Manel Kape": {"elo": 1755, "age": 32.5, "reach": 68.0, "dmg_ratio": 1.68, "slpm": 4.85, "td": 1.10, "cardio": -2.2, "layoff": 140, "cardio_nan": False, "short_notice": False},
            "Carlos Prates": {"elo": 1780, "age": 33.2, "reach": 78.0, "dmg_ratio": 1.90, "slpm": 5.50, "td": 0.20, "cardio": 1.1, "layoff": 90, "cardio_nan": False, "short_notice": False},
            "Michael Morales": {"elo": 1790, "age": 26.9, "reach": 79.0, "dmg_ratio": 1.75, "slpm": 5.20, "td": 1.40, "cardio": 1.5, "layoff": 100, "cardio_nan": False, "short_notice": False}
        }

roster = load_fighter_database()
fighter_list = list(roster.keys())

# Sidebar Controls
st.sidebar.title("📈 Portfolio & Mode Controls")
mode = st.sidebar.radio("Terminal Mode", ["Pre-Market (Model Prob Only)", "Live Trading (With Polymarket Prices)"])

bankroll = st.sidebar.number_input("USDC Bankroll ($)", min_value=10.0, value=1000.0, step=50.0)
kelly_mult = st.sidebar.selectbox("Kelly Multiplier", [0.25, 0.50, 0.125], index=0)
min_edge = st.sidebar.slider("Min Absolute Edge (%)", 1.0, 15.0, 4.0, 0.5) / 100.0
max_bet_cap = st.sidebar.slider("Max Exposure (%)", 1.0, 15.0, 5.0, 0.5) / 100.0

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
    return 0.5 * (p_fwd + (1.0 - p_rev)), 0.5 * (p_rev + (1.0 - p_fwd))

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

st.title("🎯 Verdikt UFC Quant Terminal")
st.markdown(f"**Current Mode:** `{mode}` — Search any fighter from your database and adjust metrics.")

col1, col2 = st.columns(2)

def render_fighter_search_column(col, corner_label: str, default_idx: int):
    with col:
        st.subheader(corner_label)
        selected_name = st.selectbox(f"Search Roster ({corner_label})", fighter_list, index=min(default_idx, len(fighter_list)-1), key=f"search_{corner_label}")
        
        preset = roster.get(selected_name, {"elo": 1700.0, "age": 29.0, "reach": 70.0, "dmg_ratio": 1.30, "slpm": 4.50, "td": 1.50, "cardio": 0.0, "layoff": 150.0, "cardio_nan": False, "short_notice": False})
        
        share_price = 50.0
        if mode == "Live Trading (With Polymarket Prices)":
            share_price = st.number_input("Polymarket YES Price (1¢ to 99¢)", min_value=1.0, max_value=99.0, value=50.0, step=1.0, key=f"price_{corner_label}")
        
        with st.expander("🛠️ Metrics & Manual Overrides", expanded=True):
            elo = st.number_input("Elo", 1000.0, 2200.0, float(preset["elo"]), key=f"elo_{corner_label}")
            age = st.number_input("Age", 18.0, 50.0, float(preset["age"]), key=f"age_{corner_label}")
            reach = st.number_input("Reach", 55.0, 88.0, float(preset["reach"]), key=f"reach_{corner_label}")
            dmg_ratio = st.number_input("Damage Ratio", 0.1, 5.0, float(preset["dmg_ratio"]), key=f"dmg_{label if 'label' in locals() else corner_label}")
            slpm = st.number_input("SLpM", 0.5, 12.0, float(preset["slpm"]), key=f"slpm_{corner_label}")
            td = st.number_input("TD / 15m", 0.0, 12.0, float(preset["td"]), key=f"td_{corner_label}")
            
            cardio_nan = st.checkbox("Untested R3 Cardio", value=bool(preset["cardio_nan"]), key=f"cnan_{corner_label}")
            cardio = st.number_input("Cardio Delta", -20.0, 20.0, float(preset["cardio"]), disabled=cardio_nan, key=f"card_{corner_label}")
            layoff = st.number_input("Layoff (Days)", 7.0, 1500.0, float(preset["layoff"]), key=f"lay_{corner_label}")
            
        sn = st.checkbox("Short-Notice (<14 Days)", value=bool(preset["short_notice"]), key=f"sn_{corner_label}")
        bs = st.checkbox("Missed Weight / Drain", value=False, key=f"bs_{corner_label}")
        chaos_mod = (-0.28 if sn else 0.0) + (-0.22 if bs else 0.0)
        
        return {
            "name": selected_name, "share_price": share_price, "elo": elo, "age": age, 
            "reach": reach, "dmg_ratio": dmg_ratio, "slpm": slpm, "td": td, 
            "cardio": cardio, "cardio_nan": cardio_nan, "layoff": layoff, "chaos_mod": chaos_mod
        }

f1 = render_fighter_search_column(col1, "Fighter A (Red Corner)", default_idx=0)
f2 = render_fighter_search_column(col2, "Fighter B (Blue Corner)", default_idx=1)
st.divider()

p_f1, p_f2 = predict_symmetric_probability(f1, f2)

if mode == "Pre-Market (Model Prob Only)":
    st.subheader("📊 True Model Probability Target")
    m1, m2 = st.columns(2)
    m1.metric(f"{f1['name']} True Win Probability", f"{p_f1*100:.1f}%", f"Fair Value: {p_f1*100:.0f}¢")
    m2.metric(f"{f2['name']} True Win Probability", f"{p_f2*100:.1f}%", f"Fair Value: {p_f2*100:.0f}¢")
    
    st.info(f"💡 **Pre-Market Tip:** Look for shares priced below **{p_f1*100:.0f}¢** for {f1['name']} or below **{p_f2*100:.0f}¢** for {f2['name']} to secure an edge.")
else:
    trade_f1 = calculate_polymarket_kelly(p_f1, f1["share_price"], bankroll, kelly_mult, max_bet_cap)
    trade_f2 = calculate_polymarket_kelly(p_f2, f2["share_price"], bankroll, kelly_mult, max_bet_cap)

    m1, m2 = st.columns(2)
    m1.metric(f"{f1['name']} ({f1['share_price']:.0f}¢)", f"{p_f1*100:.1f}% Model Prob", f"{trade_f1['edge']*100:+.1f}% Edge")
    m2.metric(f"{f2['name']} ({f2['share_price']:.0f}¢)", f"{p_f2*100:.1f}% Model Prob", f"{trade_f2['edge']*100:+.1f}% Edge")

    if trade_f1["edge"] >= min_edge and trade_f1["edge"] > trade_f2["edge"]:
        st.success(f"### 🟢 BUY YES: {f1['name'].upper()} @ {f1['share_price']:.0f}¢ — **USDC ${trade_f1['investment']:,.2f}** (~{trade_f1['shares']} shares)")
    elif trade_f2["edge"] >= min_edge and trade_f2["edge"] > trade_f1["edge"]:
        st.success(f"### 🟢 BUY YES: {f2['name'].upper()} @ {f2['share_price']:.0f}¢ — **USDC ${trade_f2['investment']:,.2f}** (~{trade_f2['shares']} shares)")
    else:
        st.warning("### ⚪ HOLD / PASS — Market is Efficient")
