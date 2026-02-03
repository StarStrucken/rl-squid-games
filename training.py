from typing import Optional, Callable
import gymnasium as gym
import numpy as np
import torch


def discounted_returns(rewards, gamma=0.99):
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return np.array(returns, dtype=np.float32)


def training_loop(
    env: gym.Env,
    model,
    action_function: Optional[Callable] = None,
    episodes: int = 1000,
    gamma: float = 0.99,
    train_epochs: int = 1,
    batch_episodes: int = 5,          # <-- batch size
    eps_start: float = 0.10,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
    verbose_every: int = 25,
):
    eps = eps_start

    # running flags
    hit_time_limit_once = False

    # batch buffers
    batch_states = []
    batch_actions = []
    batch_returns = []

    for ep in range(1, episodes + 1):
        s, _ = env.reset()
        done = False
        truncated = False

        traj_s, traj_a, traj_r = [], [], []

        # diagnostics
        steps = 0
        total_reward = 0.0
        shots = 0
        failed_shots = 0

        while not (done or truncated):
            state = torch.from_numpy(np.expand_dims(s, axis=0)).float()
            policy, _ = model(state)   # policy shape (1, A)
            p = policy[0]

            # soft ammo suppression
            if s[2] < 0.10:
                p = p.clone()
                p[9] *= 0.05

            p = torch.clamp(p, min=1e-8)
            p = p / p.sum()

            action = torch.multinomial(p, 1).item()

            traj_s.append(s)
            traj_a.append(action)

            s, r, done, truncated, info = env.step(action)
            traj_r.append(r)

            shots += int(info.get("shotBullet", False))
            failed_shots += int(info.get("failedShot", False))
            steps += 1
            total_reward += float(r)

        # if we ever hit the time limit, commit more (reduce eps)
        if steps >= 1800:
            hit_time_limit_once = True
        if hit_time_limit_once:
            eps = min(eps, 0.01)

        # compute per-trajectory returns (does not normalize yet) 
        returns = discounted_returns(traj_r, gamma)

        # add this episode into batch buffers
        batch_states.append(np.array(traj_s))
        batch_actions.append(np.array(traj_a))
        batch_returns.append(returns)

        # decay exploration each episode
        eps = max(eps_end, eps * eps_decay)

        # update every batch_episodes episodes
        if ep % batch_episodes == 0:
            S = np.concatenate(batch_states, axis=0)
            A = np.concatenate(batch_actions, axis=0)
            R = np.concatenate(batch_returns, axis=0)

            # normalize across the whole batch (variance reduction)
            R = (R - R.mean()) / (R.std() + 1e-8)

            model.custom_train(S, A, R, epochs=train_epochs)

            # clear batch buffers
            batch_states.clear()
            batch_actions.clear()
            batch_returns.clear()

        # testing logs
        if ep % verbose_every == 0:
            print(
                f"ep={ep:4d} "
                f"steps={steps:4d} "
                f"R={total_reward:7.2f} "
                f"shots={shots:3d} "
                f"failed={failed_shots:3d} "
                f"eps={eps:.3f} "
                f"(batch={batch_episodes})"
            )
