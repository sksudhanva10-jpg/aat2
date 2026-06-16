# CityMind AI — SDG 11 Sustainable Cities Agent

An AI agent that learns optimal resource allocation across city zones using **Q-Learning** (Reinforcement Learning).

## What It Does
- **Environment**: 5 city zones × 4 stress levels = 20 states
- **Actions**: Deploy Transport / Waste Management / Upgrade Housing / Green Spaces / Road Safety
- **Learning**: Agent starts random, gets reward/penalty per decision, and converges to an optimal policy
- **Dashboard**: Live learning curve, Q-Table heatmap, policy table, decision log

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect repo → set `app.py` as entry point → Deploy

## Deploy to Railway
1. Push to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. It auto-detects the Procfile → live in 2 minutes

## Project Structure
```
sdg11_city_agent/
├── app.py            # Streamlit dashboard
├── agent.py          # Q-Learning agent + environment
├── requirements.txt
├── Procfile          # For Railway/Render
└── README.md
```

## How to Explain It (Viva)
- **States**: Each city zone has a stress level (Low/Moderate/High/Critical)
- **Actions**: Agent picks the best intervention for that state
- **Reward**: +10 for correct action, +5 partial, -5 wrong
- **Learning**: Q-Table stores the value of every (state, action) pair — this IS the memory
- **Convergence**: Epsilon decays from 1.0 → 0.05, meaning agent stops exploring and starts exploiting learned knowledge
