# 🦑 RL Squid Games — A Stable Actor-Critic RL Agent for "Squid Hunt"
**Top 5 [ArenaX Labs 2026 Hackathon](https://competesai.com/competitions/cmp_IUpW4lf8ncP6) Submission**

---

## What
A **principled reinforcement learning agent** for *Squid Hunt*, an arcade survival game that requires:
- precise navigation
- timing-based combat
- ammo and risk management

The agent consistently achieves **long survival times (often the 30s limit)** with **stable convergence**, without reward hacks or brittle heuristics.

---

## Why 
Squid Hunt combines:
- sparse + delayed rewards  
- high-risk actions (shooting wastes ammo if mistimed)
- fast dynamics where random exploration is deadly  

Naive RL approaches (ε-greedy, REINFORCE) collapse or oscillate.

---

## Core Idea
We use a **proper Actor–Critic (policy gradient + baseline)** architecture:

- **Policy head** learns *what to do*
- **Value head** learns *how well the state is going*
- Training uses **advantages**  

This dramatically reduces variance and prevents policy regression after good episodes.

---

## Key Technical Choices (Why It Works)

### ✅ Actor–Critic with Baseline
- Eliminates REINFORCE instability
- Preserves good strategies once discovered

### ✅ No ε-Greedy Exploration
- Actions sampled directly from the policy
- Exploration driven by **entropy regularization**
- Mathematically correct for policy gradients

### ✅ Ammo-Aware Shooting (No Reward Hacking)
```python
if ammo < 0.10:
  p[SHOOT] *= 0.05
```

### Tech Stack
- Python, PyTorch, uv
- Gymnasium
- SAI Reinforcement Learning Framework
