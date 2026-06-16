"""
SDG 11 — Sustainable Cities AI Agent
Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, time, random

from agent import QLearningAgent, CityEnvironment, train_agent, ZONES, STRESS_LEVELS, ACTIONS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CityMind AI — SDG 11",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0a0f1e; }
    .block-container { padding: 2rem 2.5rem; }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem; font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1rem; color: #8892a4; margin-bottom: 2rem;
        font-weight: 300; letter-spacing: 0.02em;
    }

    .kpi-card {
        background: linear-gradient(135deg, #141b2d, #1a2340);
        border: 1px solid #1e2d4a;
        border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 24px rgba(0,212,255,0.06);
    }
    .kpi-label { font-size: 0.72rem; color: #5a6a80; text-transform: uppercase; letter-spacing: 0.1em; }
    .kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #00d4ff; }
    .kpi-delta { font-size: 0.78rem; color: #4caf8a; margin-top: 0.2rem; }

    .section-label {
        font-size: 0.7rem; font-weight: 600; color: #00d4ff;
        text-transform: uppercase; letter-spacing: 0.15em;
        margin-bottom: 0.5rem;
    }

    .decision-row {
        background: #111827; border-left: 3px solid #7b2ff7;
        border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 0.4rem;
        font-size: 0.82rem; color: #c9d4e0;
    }
    .reward-pos { color: #4caf8a; font-weight: 600; }
    .reward-neg { color: #f56565; font-weight: 600; }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.8rem; font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem; cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .stSlider > div > div { color: #00d4ff; }

    div[data-testid="stSidebar"] {
        background: #0d1526;
        border-right: 1px solid #1e2d4a;
    }

    .stress-critical { color: #f56565; font-weight: 600; }
    .stress-high     { color: #f6ad55; font-weight: 600; }
    .stress-moderate { color: #68d391; font-weight: 600; }
    .stress-low      { color: #76e4f7; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent        = QLearningAgent()
    st.session_state.trained      = False
    st.session_state.total_ep     = 0
    st.session_state.env          = CityEnvironment()
    st.session_state.last_rewards = []

agent = st.session_state.agent
env   = st.session_state.env

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏙️ CityMind AI")
    st.markdown("<div style='color:#5a6a80;font-size:0.8rem;margin-bottom:1.5rem'>SDG 11 · Sustainable Cities</div>", unsafe_allow_html=True)

    st.markdown("**Training Config**")
    episodes = st.slider("Episodes per Run", 50, 500, 100, 50)
    steps    = st.slider("Steps per Episode", 10, 50, 20, 5)

    st.markdown("---")
    train_btn = st.button("▶ Train Agent", use_container_width=True)
    reset_btn = st.button("↺ Reset Agent", use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-label'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#5a6a80;font-size:0.78rem;line-height:1.6'>
    Agent learns to allocate city resources across 5 zones × 4 stress levels
    using <b style='color:#00d4ff'>Q-Learning</b>.<br><br>
    It improves with each episode — starting random, converging to optimal policy.
    </div>
    """, unsafe_allow_html=True)

# ── Reset ─────────────────────────────────────────────────────────────────────
if reset_btn:
    st.session_state.agent    = QLearningAgent()
    st.session_state.trained  = False
    st.session_state.total_ep = 0
    st.session_state.last_rewards = []
    agent = st.session_state.agent
    st.rerun()

# ── Train ─────────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner(f"Training for {episodes} episodes..."):
        agent, env, rewards = train_agent(agent, episodes=episodes, steps=steps)
        st.session_state.agent        = agent
        st.session_state.env          = env
        st.session_state.trained      = True
        st.session_state.total_ep    += episodes
        st.session_state.last_rewards = rewards
    st.success(f"✓ Trained {episodes} episodes. Total: {st.session_state.total_ep}")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>CityMind AI Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>SDG 11 · Sustainable Cities & Communities · Reinforcement Learning Dashboard</div>", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

total_ep     = st.session_state.total_ep
history      = agent.history
avg_reward   = round(np.mean([h["reward"] for h in history[-100:]]), 1) if history else 0
epsilon_now  = round(agent.epsilon, 3)
decisions_ct = len(agent.decisions)

with k1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Episodes Trained</div>
        <div class='kpi-value'>{total_ep}</div>
        <div class='kpi-delta'>↑ lifetime total</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Avg Reward (last 100)</div>
        <div class='kpi-value'>{avg_reward}</div>
        <div class='kpi-delta'>{'↑ improving' if avg_reward > 0 else '— not trained'}</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Exploration Rate ε</div>
        <div class='kpi-value'>{epsilon_now}</div>
        <div class='kpi-delta'>{'↓ converging' if epsilon_now < 0.5 else '↑ still exploring'}</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Total Decisions Made</div>
        <div class='kpi-value'>{decisions_ct}</div>
        <div class='kpi-delta'>↑ experience accumulated</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts Row ────────────────────────────────────────────────────────────────
if history:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='section-label'>Learning Curve — Reward per Episode</div>", unsafe_allow_html=True)
        df_hist = pd.DataFrame(history)

        # Rolling average
        df_hist["rolling"] = df_hist["reward"].rolling(10, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["episode"], y=df_hist["reward"],
            mode="lines", name="Episode Reward",
            line=dict(color="#1e3a5f", width=1), opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=df_hist["episode"], y=df_hist["rolling"],
            mode="lines", name="10-ep Moving Avg",
            line=dict(color="#00d4ff", width=2.5)
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892a4", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            xaxis=dict(gridcolor="#1a2340", title="Episode"),
            yaxis=dict(gridcolor="#1a2340", title="Total Reward"),
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-label'>Exploration vs Exploitation</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_hist["episode"], y=df_hist["epsilon"],
            mode="lines", fill="tozeroy",
            line=dict(color="#7b2ff7", width=2),
            fillcolor="rgba(123,47,247,0.15)",
            name="Epsilon (ε)"
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892a4", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="#1a2340", title="Episode"),
            yaxis=dict(gridcolor="#1a2340", title="ε value", range=[0, 1.05]),
            showlegend=False, height=280
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Q-Table Heatmap ───────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Q-Table Heatmap — Agent's Learned Value for Each (State, Action) Pair</div>", unsafe_allow_html=True)

    state_labels = [f"{z[:8]}·{s}" for z in ZONES for s in STRESS_LEVELS]
    action_labels = [a.replace(" ", "\n") for a in ACTIONS]

    fig3 = go.Figure(data=go.Heatmap(
        z=agent.q_table,
        x=ACTIONS,
        y=state_labels,
        colorscale=[[0, "#0a0f1e"], [0.5, "#1e3a8a"], [1, "#00d4ff"]],
        showscale=True,
        hoverongaps=False,
        hovertemplate="State: %{y}<br>Action: %{x}<br>Q-Value: %{z:.2f}<extra></extra>"
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4", size=10),
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("👈 Hit **Train Agent** in the sidebar to start learning.")

# ── Policy Table ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Learned Policy — Best Action per Zone × Stress Level</div>", unsafe_allow_html=True)

policy = agent.get_policy(env)
df_policy = pd.DataFrame(policy)

stress_color = {"Low": "🟦", "Moderate": "🟩", "High": "🟧", "Critical": "🟥"}
df_policy["Stress"] = df_policy["stress"].map(lambda s: f"{stress_color.get(s,'')} {s}")
df_policy = df_policy.rename(columns={
    "zone": "Zone", "action": "Recommended Action", "q_value": "Q-Value (Confidence)"
})[["Zone", "Stress", "Recommended Action", "Q-Value (Confidence)"]]

st.dataframe(
    df_policy,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Q-Value (Confidence)": st.column_config.ProgressColumn(
            "Q-Value (Confidence)", min_value=0, max_value=110, format="%.1f"
        )
    }
)

# ── Recent Decisions ──────────────────────────────────────────────────────────
if agent.decisions:
    st.markdown("<br><div class='section-label'>Recent Agent Decisions</div>", unsafe_allow_html=True)
    last = agent.decisions[-10:][::-1]
    for d in last:
        color_class = "reward-pos" if d["reward"] >= 0 else "reward-neg"
        reward_str  = f"+{d['reward']}" if d["reward"] >= 0 else str(d["reward"])
        st.markdown(f"""
        <div class='decision-row'>
            📍 <b>{d['zone']}</b> &nbsp;·&nbsp; Stress: <b>{d['stress']}</b>
            &nbsp;→&nbsp; 🏗 {d['action']}
            &nbsp;&nbsp;<span class='{color_class}'>{reward_str}</span>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='color:#2d3a4f;font-size:0.75rem;text-align:center'>
    CityMind AI · Built with Q-Learning · SDG 11 Sustainable Cities & Communities
</div>""", unsafe_allow_html=True)
