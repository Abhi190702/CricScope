import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from model.feature_engineering import (
    compute_venue_chase_win_rate,
    compute_toss_win_rate,
    compute_team_toss_win_rate,
    get_venue_batting_style,
)
import logging

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CricScope",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; color: #e2dfd8; }

[data-testid="stAppViewContainer"] {
    background: #080808;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,175,55,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,90,30,0.05) 0%, transparent 50%);
    min-height: 100vh;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

section[data-testid="stSidebar"] {
    background: #0c0c0c;
    border-right: 1px solid rgba(212,175,55,0.12);
    width: 260px !important;
}
section[data-testid="stSidebar"] > div { padding: 0; }

.sidebar-brand {
    padding: 36px 28px 24px;
    border-bottom: 1px solid rgba(212,175,55,0.1);
    margin-bottom: 16px;
}
.sidebar-logo-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px; font-weight: 600; letter-spacing: 3px;
    background: linear-gradient(135deg, #f0d060 0%, #d4af37 40%, #a07820 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; display: block; margin-bottom: 4px;
}
.sidebar-tagline {
    font-size: 10px; letter-spacing: 2.5px; text-transform: uppercase;
    color: rgba(212,175,55,0.45); font-weight: 400;
}
.sidebar-section-label {
    font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
    color: rgba(180,160,100,0.35); padding: 12px 28px 6px; font-weight: 500;
}

.stButton > button {
    width: 100%; text-align: left; background: transparent; border: none;
    border-radius: 0; color: rgba(220,210,180,0.65);
    font-family: 'DM Sans', sans-serif; font-size: 13px;
    font-weight: 400; letter-spacing: 0.5px; padding: 11px 28px;
    height: auto; transition: all 0.2s ease;
}
.stButton > button:hover  { background: rgba(212,175,55,0.06); color: #d4af37; border: none; box-shadow: none; }
.stButton > button:active,
.stButton > button:focus  { background: rgba(212,175,55,0.1);  color: #f0d060; border: none; box-shadow: none; outline: none; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* hero */
.hero-wrapper {
    padding: 64px 60px 40px;
    border-bottom: 1px solid rgba(212,175,55,0.08);
    position: relative; overflow: hidden;
}
.hero-wrapper::before {
    content: ''; position: absolute; top: -60px; left: -60px; right: -60px; height: 200px;
    background: radial-gradient(ellipse, rgba(212,175,55,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow  { font-size: 10px; letter-spacing: 4px; text-transform: uppercase; color: rgba(212,175,55,0.5); margin-bottom: 18px; font-weight: 400; }
.hero-title    { font-family: 'Cormorant Garamond', serif; font-size: clamp(52px,7vw,88px); font-weight: 600; line-height: 0.95; letter-spacing: -1px; background: linear-gradient(160deg, #ffffff 0%, #f8f0d0 30%, #d4af37 70%, #a07820 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 18px; }
.hero-subtitle { font-size: 15px; color: rgba(220,210,185,0.55); font-weight: 300; letter-spacing: 0.3px; max-width: 460px; line-height: 1.6; }
.hero-badge    { display: inline-flex; align-items: center; gap: 7px; background: rgba(212,175,55,0.08); border: 1px solid rgba(212,175,55,0.2); border-radius: 100px; padding: 5px 14px 5px 10px; font-size: 11px; color: rgba(212,175,55,0.8); letter-spacing: 0.5px; margin-bottom: 24px; width: fit-content; }
.hero-dot      { width: 6px; height: 6px; border-radius: 50%; background: #d4af37; animation: pulse-dot 2s infinite; }
@keyframes pulse-dot { 0%, 100% { opacity:1; transform:scale(1); } 50% { opacity:0.5; transform:scale(0.8); } }

/* stats row */
.stats-row { display: flex; gap: 16px; padding: 24px 60px; border-bottom: 1px solid rgba(212,175,55,0.06); }
.stat-pill  { flex: 1; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 18px 22px; transition: all 0.25s ease; }
.stat-pill:hover { background: rgba(212,175,55,0.04); border-color: rgba(212,175,55,0.15); transform: translateY(-1px); }
.stat-value { font-family: 'DM Mono', monospace; font-size: 26px; font-weight: 500; color: #e8d89a; line-height: 1; margin-bottom: 6px; }
.stat-label { font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: rgba(200,185,140,0.4); }

/* input cards */
.input-card       { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07); border-radius: 20px; padding: 28px 32px; backdrop-filter: blur(20px); transition: border-color 0.3s ease; }
.input-card:hover { border-color: rgba(212,175,55,0.15); }
.input-label      { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: rgba(212,175,55,0.5); margin-bottom: 12px; font-weight: 500; }

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2dfd8 !important;
}
.stSelectbox label, .stNumberInput label, .stSlider label, .stTextInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 10px !important; letter-spacing: 1.8px !important;
    text-transform: uppercase !important; color: rgba(200,185,140,0.5) !important;
    font-weight: 500 !important;
}

/* analyse button */
.stButton.analyze-btn > button {
    background: linear-gradient(135deg, #c9a227 0%, #d4af37 40%, #e8c84a 100%);
    color: #0a0800; border: none; border-radius: 14px; height: 52px;
    font-family: 'DM Sans', sans-serif; font-size: 13px; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase; transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(212,175,55,0.2); width: 100%;
}
.stButton.analyze-btn > button:hover {
    box-shadow: 0 12px 48px rgba(212,175,55,0.35), 0 0 60px rgba(212,175,55,0.1);
    transform: translateY(-2px); filter: brightness(1.05);
    color: #0a0800; border: none;
}

/* prediction card */
.prediction-card { background: rgba(212,175,55,0.04); border: 1px solid rgba(212,175,55,0.18); border-radius: 24px; padding: 36px 32px; position: relative; overflow: hidden; }
.prediction-card::before { content: ''; position: absolute; top: -1px; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, #d4af37, transparent); }
.prediction-card::after  { content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(ellipse 70% 60% at 50% 0%, rgba(212,175,55,0.06) 0%, transparent 60%); pointer-events: none; }

.prediction-label  { font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: rgba(212,175,55,0.4); margin-bottom: 24px; font-weight: 500; }
.win-probability   { font-family: 'DM Mono', monospace; font-size: 72px; font-weight: 500; background: linear-gradient(135deg, #f0d060, #d4af37); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; margin-bottom: 4px; }
.win-prob-label    { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: rgba(200,185,140,0.35); margin-bottom: 28px; }
.prob-bar-track    { height: 6px; background: rgba(255,255,255,0.05); border-radius: 100px; overflow: hidden; }
.prob-bar-fill     { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #b8962e, #d4af37, #f0d060); transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1); box-shadow: 0 0 12px rgba(212,175,55,0.4); }
.prob-bar-labels   { display: flex; justify-content: space-between; margin-top: 10px; font-size: 11px; color: rgba(200,185,140,0.4); font-family: 'DM Mono', monospace; }
.metrics-row       { display: flex; gap: 10px; margin-top: 18px; }
.metric-chip       { flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 12px 14px; text-align: center; }
.metric-chip-value { font-family: 'DM Mono', monospace; font-size: 16px; color: #d4c080; font-weight: 500; margin-bottom: 4px; }
.metric-chip-label { font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: rgba(180,165,115,0.35); }

.stProgress > div > div { background: linear-gradient(90deg,#b8962e,#d4af37) !important; border-radius: 100px !important; }
.stProgress > div       { background: rgba(255,255,255,0.04) !important; border-radius: 100px !important; height: 6px !important; }

div[data-testid="metric-container"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 16px 20px; }
div[data-testid="metric-container"] label { color: rgba(200,185,140,0.45) !important; font-size: 10px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; color: #e8d89a !important; font-size: 28px !important; }

hr { border: none; border-top: 1px solid rgba(212,175,55,0.08); margin: 0; }
.main-pad { padding: 0 60px 60px; }
::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-track { background: #0c0c0c; }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.25); border-radius: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# TEAM DATA
# ─────────────────────────────────────────────
team_data = {
    "Chennai Super Kings":        {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/1-5.jpg",                                                                                   "abbr": "CSK", "color": "#facc15"},
    "Delhi Capitals":             {"logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/2-4.jpg",               "abbr": "DC",  "color": "#3b82f6"},
    "Punjab Kings":               {"logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/5-4.jpg",               "abbr": "PBKS","color": "#ef4444"},
    "Kolkata Knight Riders":      {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/3-4.jpg",                                                                                   "abbr": "KKR", "color": "#7c3aed"},
    "Mumbai Indians":             {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/4-4.jpg",                                                                                   "abbr": "MI",  "color": "#3b82f6"},
    "Rajasthan Royals":           {"logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/6-4.jpg",               "abbr": "RR",  "color": "#ec4899"},
    "Royal Challengers Bangalore":{"logo": "https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/Untitled-4.jpg",                                                                          "abbr": "RCB", "color": "#dc2626"},
    "Sunrisers Hyderabad":        {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/8-4.jpg",                                                                                   "abbr": "SRH", "color": "#f97316"},
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_model(model_name="logistic"):
    return {
        "logistic":      LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(),
        "xgboost":       XGBClassifier(),
    }[model_name]


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(model_name="logistic"):
    model_path = f"{model_name}_model.pkl"

    if os.path.exists(model_path):
        data = joblib.load(model_path)
        return data["pipe"], data["venue_chase_rates"], data["toss_win_rates"]

    matches    = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    venue_chase_rates = compute_venue_chase_win_rate(matches)
    toss_win_rates    = compute_toss_win_rate(matches)

    df = deliveries.merge(matches, left_on="match_id", right_on="id")

    total_df = (
        df[df["inning"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "target"})
    )
    df = df.merge(total_df, on="match_id")
    df = df[df["inning"] == 2]

    df["current_score"]    = df.groupby("match_id")["total_runs"].cumsum()
    df["runs_left"]        = df["target"] - df["current_score"]
    df["balls_left"]       = 120 - (df["over"] * 6 + df["ball"])
    df["player_dismissed"] = df["player_dismissed"].notna().astype(int)
    df["wickets"]          = 10 - df.groupby("match_id")["player_dismissed"].cumsum()

    balls_played   = (120 - df["balls_left"]).replace(0, 1)
    df["crr"]      = (df["current_score"] * 6) / balls_played
    df["rrr"]      = (df["runs_left"] * 6) / df["balls_left"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    global_chase_avg = sum(venue_chase_rates.values()) / len(venue_chase_rates)
    global_toss_avg  = sum(toss_win_rates.values())    / len(toss_win_rates)

    df["venue_chase_win_rate"] = df["venue"].map(venue_chase_rates).fillna(global_chase_avg)
    df["toss_win_rate"]        = df["venue"].map(toss_win_rates).fillna(global_toss_avg)
    df["result"]               = np.where(df["batting_team"] == df["winner"], 1, 0)

    final_df = df[[
        "batting_team", "bowling_team", "city",
        "runs_left", "balls_left", "wickets", "target", "crr", "rrr",
        "venue_chase_win_rate", "toss_win_rate", "result",
    ]].dropna()

    X = final_df.drop("result", axis=1)
    y = final_df["result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["batting_team", "bowling_team", "city"]),
        ("num", "passthrough", ["runs_left", "balls_left", "wickets", "target", "crr", "rrr",
                                "venue_chase_win_rate", "toss_win_rate"]),
    ])

    pipe = Pipeline([("preprocessor", preprocessor), ("model", get_model(model_name))])

    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    logging.info(f"CV Scores: {scores}  |  Avg: {scores.mean():.4f}")

    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    logging.info(f"Test Accuracy: {acc:.4f}")

    joblib.dump({"pipe": pipe, "venue_chase_rates": venue_chase_rates, "toss_win_rates": toss_win_rates}, model_path)
    return pipe, venue_chase_rates, toss_win_rates


pipe, venue_chase_rates, toss_win_rates = train_model()

matches_df          = pd.read_csv("matches.csv")
venue_list          = sorted(matches_df["venue"].dropna().unique().tolist())
city_list           = sorted(matches_df["city"].dropna().unique().tolist())
team_toss_win_rates = compute_team_toss_win_rate(matches_df)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="sidebar-logo-text">CRICSCOPE</span>
            <span class="sidebar-tagline">Match Intelligence Platform</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

    if st.button("◈  Dashboard",     key="nav_dash"):     st.session_state.page = "Dashboard"
    if st.button("◉  Match Analysis", key="nav_analysis"): st.session_state.page = "Analysis"

    st.markdown('<div style="height:1px;background:rgba(212,175,55,0.08);margin:16px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Built By</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="padding:0 18px 8px;">
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(212,175,55,0.12);border-radius:16px;padding:20px 18px 14px;position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:60px;background:radial-gradient(ellipse at 50% 0%,rgba(212,175,55,0.08) 0%,transparent 70%);pointer-events:none;"></div>
                <div style="width:44px;height:44px;border-radius:50%;background:linear-gradient(135deg,#c9a227,#f0d060);display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;color:#0a0800;margin-bottom:12px;box-shadow:0 0 18px rgba(212,175,55,0.25);">AS</div>
                <div style="font-size:17px;font-weight:600;color:#f0e8cc;letter-spacing:0.5px;margin-bottom:3px;">Arnav Singh</div>
                <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:rgba(212,175,55,0.4);margin-bottom:18px;font-weight:500;">ML · Data · Analytics</div>
                <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(212,175,55,0.15),transparent);margin-bottom:12px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="padding:0 18px;">
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(212,175,55,0.12);border-top:none;border-radius:0 0 16px 16px;padding:4px 10px 14px;">
                <p style="margin:0 0 2px;padding:8px 8px;">
                    <span style="color:rgba(212,175,55,0.6);margin-right:8px;font-size:12px;">✉</span>
                    <a href="mailto:itsarnav.singh80@gmail.com" style="color:rgba(200,185,140,0.6);font-size:11px;text-decoration:none;">itsarnav.singh80@gmail.com</a>
                </p>
                <p style="margin:0 0 2px;padding:8px 8px;">
                    <span style="color:rgba(212,175,55,0.6);margin-right:8px;font-size:12px;">in</span>
                    <a href="https://www.linkedin.com/in/arnav-singh-a87847351" target="_blank" style="color:rgba(200,185,140,0.6);font-size:11px;text-decoration:none;">linkedin.com/in/arnav-singh</a>
                </p>
                <p style="margin:0;padding:8px 8px;">
                    <span style="color:rgba(212,175,55,0.6);margin-right:8px;font-size:12px;">&#9670;</span>
                    <a href="https://github.com/Arnav-Singh-5080" target="_blank" style="color:rgba(200,185,140,0.6);font-size:11px;text-decoration:none;">Arnav-Singh-5080</a>
                </p>
            </div>
        </div>
        <div style="text-align:center;margin-top:16px;padding-bottom:24px;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(200,185,140,0.18);">
            CricScope v2.0 · IPL Edition
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────
if st.session_state.page == "Dashboard":

    st.markdown(
        """
        <div class="hero-wrapper">
            <div class="hero-eyebrow">IPL Match Intelligence · Season 2025</div>
            <div class="hero-badge"><div class="hero-dot"></div>Live Predictions Active</div>
            <div class="hero-title">CricScope</div>
            <div class="hero-subtitle">Precision match analytics engineered for modern cricket. Real-time win probability powered by machine learning.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stats-row">
            <div class="stat-pill"><div class="stat-value">8</div><div class="stat-label">IPL Teams</div></div>
            <div class="stat-pill"><div class="stat-value">ML</div><div class="stat-label">Model Type</div></div>
            <div class="stat-pill"><div class="stat-value">120</div><div class="stat-label">Balls Tracked</div></div>
            <div class="stat-pill"><div class="stat-value">8+</div><div class="stat-label">Key Signals</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="padding:48px 60px 24px;">
            <div style="font-family:'Cormorant Garamond',serif;font-size:13px;letter-spacing:3px;text-transform:uppercase;color:rgba(212,175,55,0.4);margin-bottom:28px;">IPL Teams</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    team_cols = st.columns(4)
    for i, (team_name, tdata) in enumerate(team_data.items()):
        tc     = tdata["color"]
        abbr   = tdata["abbr"]
        logo   = tdata["logo"]
        with team_cols[i % 4]:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:20px;text-align:center;margin-bottom:12px;">'
                f'<div style="width:72px;height:72px;border-radius:50%;margin:0 auto;overflow:hidden;background:#111;box-shadow:0 0 20px {tc}50;display:flex;align-items:center;justify-content:center;">'
                f'<img src="{logo}" style="width:100%;height:100%;object-fit:cover;mix-blend-mode:screen;border-radius:50%;" /></div>'
                f'<div style="font-family:\'Cormorant Garamond\',serif;font-size:18px;font-weight:600;color:{tc};letter-spacing:2px;margin-top:12px;">{abbr}</div>'
                f'<div style="font-size:10px;color:rgba(200,185,140,0.35);margin-top:4px;">{team_name}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div style="padding:0 60px 32px;text-align:center;">
            <div style="display:inline-block;background:rgba(212,175,55,0.06);border:1px solid rgba(212,175,55,0.15);border-radius:14px;padding:20px 36px;">
                <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:rgba(212,175,55,0.5);margin-bottom:8px;">Get Started</div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:20px;color:#f0e8cc;font-weight:500;">Open Match Analysis from the sidebar →</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# ANALYSIS PAGE
# ─────────────────────────────────────────────
if st.session_state.page == "Analysis":

    st.markdown(
        """
        <div class="hero-wrapper" style="padding-bottom:32px;">
            <div class="hero-eyebrow">Win Probability Engine</div>
            <div class="hero-title" style="font-size:clamp(36px,4vw,56px);margin-bottom:10px;">Match Analysis</div>
            <div class="hero-subtitle">Configure the match state below to compute real-time win probabilities.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-pad">', unsafe_allow_html=True)
    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

    teams = list(team_data.keys())

    st.markdown(
        '<div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;color:rgba(212,175,55,0.4);margin-bottom:20px;font-weight:500;">Match Configuration</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">Teams &amp; Venue</div>', unsafe_allow_html=True)
        batting_team = st.selectbox("Batting Team",  teams, key="bat")
        bowling_team = st.selectbox("Bowling Team",  [t for t in teams if t != batting_team], key="bowl")
        city         = st.selectbox("City",          city_list,  key="city")
        venue        = st.selectbox("Venue",         venue_list, key="venue")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<div class="input-label">Match State</div>', unsafe_allow_html=True)
        target  = st.number_input("Target Score",    min_value=50, max_value=300,        value=180, step=1)
        score   = st.number_input("Current Score",   min_value=0,  max_value=target - 1, value=50,  step=1)
        col_ov, col_wk = st.columns(2)
        with col_ov:
            overs   = st.slider("Overs Completed", min_value=1, max_value=19, value=10)
        with col_wk:
            wickets = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=2)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)

    # ── pre-compute all context values ──────────────────────────────────────
    t1 = team_data[batting_team]
    t2 = team_data[bowling_team]

    chase_rate   = venue_chase_rates.get(venue, 0.5)
    toss_rate    = toss_win_rates.get(venue, 0.5)
    bat_toss     = team_toss_win_rates.get(batting_team, 0.5)
    bowl_toss    = team_toss_win_rates.get(bowling_team, 0.5)

    venue_style  = get_venue_batting_style(venue_chase_rates).get(
        venue, {"style": "Neutral", "confidence": "Toss Up", "rate": 0.5}
    )

    chase_pct      = round(chase_rate * 100)
    toss_pct       = round(toss_rate  * 100)
    bat_toss_pct   = round(bat_toss   * 100)
    bowl_toss_pct  = round(bowl_toss  * 100)
    venue_short    = venue.split(" ")[0]
    t1_abbr        = t1["abbr"]
    t2_abbr        = t2["abbr"]
    t1_color       = t1["color"]
    t2_color       = t2["color"]
    style_label    = venue_style["style"]
    style_conf     = venue_style["confidence"]
    style_rate_pct = round(venue_style["rate"] * 100)
    style_color    = (
        "#4ade80" if style_label == "Chase"
        else "#f87171" if style_label == "Bat First"
        else "#e8d89a"
    )

    # ── venue context cards ──────────────────────────────────────────────────
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    card_base = "background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:16px 18px;min-height:110px;display:flex;flex-direction:column;justify-content:center;"
    lbl_style = "font-size:9px;letter-spacing:2px;text-transform:uppercase;color:rgba(212,175,55,0.35);margin-bottom:8px;"
    val_base  = "font-family:'DM Mono',monospace;font-size:26px;line-height:1;"
    sub_style = "font-size:9px;color:rgba(200,185,140,0.35);margin-top:8px;"

    cards_html = (
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;width:100%;margin-bottom:8px;">'

        f'<div style="{card_base}">'
        f'<div style="{lbl_style}">Chase Win Rate · {venue_short}</div>'
        f'<div style="{val_base}color:#e8d89a;">{chase_pct}%</div>'
        f'</div>'

        f'<div style="{card_base}">'
        f'<div style="{lbl_style}">Toss → Win Rate · {venue_short}</div>'
        f'<div style="{val_base}color:#e8d89a;">{toss_pct}%</div>'
        f'</div>'

        f'<div style="{card_base}">'
        f'<div style="{lbl_style}">{t1_abbr} Toss → Win</div>'
        f'<div style="{val_base}color:{t1_color};">{bat_toss_pct}%</div>'
        f'</div>'

        f'<div style="{card_base}">'
        f'<div style="{lbl_style}">{t2_abbr} Toss → Win</div>'
        f'<div style="{val_base}color:{t2_color};">{bowl_toss_pct}%</div>'
        f'</div>'

        f'<div style="{card_base}">'
        f'<div style="{lbl_style}">{venue_short} · Batting Style</div>'
        f'<div style="font-family:\'DM Mono\',monospace;font-size:22px;color:{style_color};line-height:1;">{style_label}</div>'
        f'<div style="{sub_style}">{style_conf} · {style_rate_pct}% chase wins</div>'
        f'</div>'

        f'</div>'
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # ── analyse button ───────────────────────────────────────────────────────
    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    analyze = st.button("Run Analysis", key="analyze_btn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── prediction output ────────────────────────────────────────────────────
    if analyze:
        runs_left  = target - score
        balls_left = 120 - (overs * 6)
        crr        = score / overs if overs > 0 else 0
        rrr        = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            "batting_team":         [batting_team],
            "bowling_team":         [bowling_team],
            "city":                 [city],
            "runs_left":            [runs_left],
            "balls_left":           [balls_left],
            "wickets":              [10 - wickets],
            "target":               [target],
            "crr":                  [crr],
            "rrr":                  [rrr],
            "venue_chase_win_rate": [chase_rate],
            "toss_win_rate":        [toss_rate],
        })

        with st.spinner(""):
            time.sleep(0.4)
            proba = pipe.predict_proba(input_df)[0]

        win  = proba[1]
        lose = proba[0]

        st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;color:rgba(212,175,55,0.4);margin-bottom:16px;font-weight:500;">Prediction Output</div>',
            unsafe_allow_html=True,
        )

        res_col1, res_col2 = st.columns(2, gap="large")

        # pre-compute display values
        bat_pct  = round(win  * 100)
        bowl_pct = round(lose * 100)
        crr_fmt  = round(crr, 2)
        rrr_fmt  = round(rrr, 2)
        in_hand  = 10 - wickets
        t1_name  = batting_team
        t2_name  = bowling_team

        with res_col1:
            st.markdown(
                f'<div class="prediction-card">'
                f'<div class="prediction-label">Batting Team · {t1_abbr}</div>'
                f'<div style="font-family:\'Cormorant Garamond\',serif;font-size:22px;font-weight:500;color:#c8b870;margin-bottom:16px;">{t1_name}</div>'
                f'<div class="win-probability">{bat_pct}%</div>'
                f'<div class="win-prob-label">Win Probability</div>'
                f'<div class="prob-bar-track"><div class="prob-bar-fill" style="width:{bat_pct}%;"></div></div>'
                f'<div class="prob-bar-labels"><span>0%</span><span>{bat_pct}%</span><span>100%</span></div>'
                f'<div class="metrics-row">'
                f'<div class="metric-chip"><div class="metric-chip-value">{score}</div><div class="metric-chip-label">Score</div></div>'
                f'<div class="metric-chip"><div class="metric-chip-value">{runs_left}</div><div class="metric-chip-label">Needed</div></div>'
                f'<div class="metric-chip"><div class="metric-chip-value">{balls_left}</div><div class="metric-chip-label">Balls Left</div></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:24px;padding:36px 32px;position:relative;overflow:hidden;">'
                f'<div class="prediction-label">Bowling Team · {t2_abbr}</div>'
                f'<div style="font-family:\'Cormorant Garamond\',serif;font-size:22px;font-weight:500;color:#c8b870;margin-bottom:16px;">{t2_name}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:72px;font-weight:500;color:rgba(200,185,140,0.55);line-height:1;margin-bottom:4px;">{bowl_pct}%</div>'
                f'<div class="win-prob-label">Win Probability</div>'
                f'<div class="prob-bar-track"><div style="height:100%;border-radius:100px;background:rgba(200,185,140,0.2);width:{bowl_pct}%;transition:width 0.8s ease;"></div></div>'
                f'<div class="prob-bar-labels"><span>0%</span><span>{bowl_pct}%</span><span>100%</span></div>'
                f'<div class="metrics-row">'
                f'<div class="metric-chip"><div class="metric-chip-value">{crr_fmt}</div><div class="metric-chip-label">CRR</div></div>'
                f'<div class="metric-chip"><div class="metric-chip-value">{rrr_fmt}</div><div class="metric-chip-label">RRR</div></div>'
                f'<div class="metric-chip"><div class="metric-chip-value">{in_hand}</div><div class="metric-chip-label">In Hand</div></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        verdict    = batting_team if win > 0.5 else bowling_team
        conf       = max(win, lose)
        conf_label = "High" if conf > 0.75 else "Moderate" if conf > 0.55 else "Close"
        conf_pct   = round(conf * 100)

        st.markdown(
            f'<div style="background:rgba(212,175,55,0.03);border:1px solid rgba(212,175,55,0.1);border-radius:16px;padding:20px 28px;display:flex;align-items:center;justify-content:space-between;">'
            f'<div>'
            f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:rgba(212,175,55,0.35);margin-bottom:6px;">Model Verdict</div>'
            f'<div style="font-family:\'Cormorant Garamond\',serif;font-size:22px;font-weight:500;color:#f0e8cc;">{verdict} favoured to win</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:rgba(212,175,55,0.35);margin-bottom:6px;">Confidence</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:20px;color:#d4af37;">{conf_label} · {conf_pct}%</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)