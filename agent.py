"""
SDG 11 - Sustainable Cities & Communities
Q-Learning Agent: learns optimal resource allocation across city zones
"""

import numpy as np
import random
import json

# ── Config ──────────────────────────────────────────────────────────────────
ZONES        = ["Central Business", "Residential North", "Industrial East", "Suburban South", "Old City"]
STRESS_LEVELS = ["Low", "Moderate", "High", "Critical"]
ACTIONS      = ["Deploy Public Transport", "Improve Waste Management",
                "Upgrade Housing", "Install Green Spaces", "Enhance Road Safety"]

# Best action per stress level (domain knowledge)
IDEAL_ACTION = {0: 3, 1: 0, 2: 1, 3: 2}   # Low→Green, Mod→Transport, High→Waste, Crit→Housing

N_STATES  = len(ZONES) * len(STRESS_LEVELS)   # 20
N_ACTIONS = len(ACTIONS)                        # 5

# ── Environment ──────────────────────────────────────────────────────────────
class CityEnvironment:
    def __init__(self):
        self.n_states  = N_STATES
        self.n_actions = N_ACTIONS
        self.reset()

    def reset(self):
        self.zone_idx   = random.randint(0, len(ZONES) - 1)
        self.stress_idx = random.randint(0, len(STRESS_LEVELS) - 1)
        return self._state()

    def _state(self):
        return self.zone_idx * len(STRESS_LEVELS) + self.stress_idx

    def step(self, action):
        ideal  = IDEAL_ACTION[self.stress_idx]
        diff   = abs(action - ideal)
        reward = {0: 10, 1: 5, 2: 1}.get(diff, -5) + random.uniform(-1, 1)

        self.zone_idx   = random.randint(0, len(ZONES) - 1)
        self.stress_idx = random.randint(0, len(STRESS_LEVELS) - 1)
        return self._state(), round(reward, 2), False

    def decode(self, state):
        return ZONES[state // len(STRESS_LEVELS)], STRESS_LEVELS[state % len(STRESS_LEVELS)]


# ── Agent ─────────────────────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.05):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table       = np.zeros((N_STATES, N_ACTIONS))
        self.history       = []          # (episode, avg_reward)
        self.decisions     = []          # last N decisions for UI

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_error  = reward + self.gamma * best_next - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self, env):
        """Return best action for every state."""
        policy = []
        for s in range(N_STATES):
            zone, stress = env.decode(s)
            best_a = int(np.argmax(self.q_table[s]))
            policy.append({
                "zone": zone,
                "stress": stress,
                "action": ACTIONS[best_a],
                "q_value": round(float(np.max(self.q_table[s])), 2)
            })
        return policy

    def to_dict(self):
        return {
            "q_table":  self.q_table.tolist(),
            "epsilon":  round(self.epsilon, 4),
            "history":  self.history,
            "decisions": self.decisions[-20:]
        }

    def from_dict(self, d):
        self.q_table   = np.array(d["q_table"])
        self.epsilon   = d["epsilon"]
        self.history   = d["history"]
        self.decisions = d["decisions"]


# ── Training helper (called from Streamlit) ───────────────────────────────────
def train_agent(agent: QLearningAgent, episodes=100, steps=20):
    env = CityEnvironment()
    ep_rewards = []

    for ep in range(episodes):
        state        = env.reset()
        total_reward = 0

        for _ in range(steps):
            action              = agent.choose_action(state)
            next_state, reward, _ = env.step(action)
            agent.learn(state, action, reward, next_state)

            zone, stress = env.decode(state)
            agent.decisions.append({
                "zone": zone, "stress": stress,
                "action": ACTIONS[action], "reward": reward
            })
            state        = next_state
            total_reward += reward

        agent.decay()
        ep_rewards.append(round(total_reward, 2))

        # Log every episode for chart
        agent.history.append({
            "episode": len(agent.history) + 1,
            "reward":  round(total_reward, 2),
            "epsilon": round(agent.epsilon, 4)
        })

    return agent, env, ep_rewards
