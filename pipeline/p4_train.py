"""
Pipeline 4 — SAC Training Loop
================================
Trains the Stage 1 SAC + TTFE agent on pre-RTC+B ERCOT data.

Run AFTER p0, p1, p2 (data ready), p3 (model shapes verified).

Usage:
    python pipeline/p4_train.py

Outputs:
    checkpoints/stage1/stage1_step<N>.pt    ← saved every SAVE_EVERY steps
    checkpoints/stage1/stage1_best.pt       ← best validation revenue
    logs/training_log.csv                   ← step-by-step metrics
"""

import os
import sys
import glob
import math
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import deque
from torch.optim import Adam
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.config import *
from pipeline.p3_models import TTFE, Actor, Critic, FeasibilityProjection


# ════════════════════════════════════════════════════════
# DATASET LOADER
# ════════════════════════════════════════════════════════

class ERCOTDataset:
    """Loads merged parquets and serves rolling price windows + obs components."""

    def __init__(self, split: str = "train"):
        """
        Args:
            split: "train" (before VAL_START) or "val" (from VAL_START onward)
        """
        self.split = split
        self.df    = self._load()
        self.n     = len(self.df)

        # Load normaliser stats saved by p2_build_dataset.py
        stats_path = os.path.join(CHECKPOINT_DIR, "normaliser_stats.npz")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Normaliser not found at {stats_path}\n"
                "Run p2_build_dataset.py first."
            )
        stats       = np.load(stats_path, allow_pickle=True)
        self.mean   = stats["mean"].astype(np.float32)   # shape: (19,) = 12+7
        self.std    = stats["std"].astype(np.float32)

        print(f"[Dataset:{split}] {self.n:,} rows | "
              f"{self.df.index.min().date()} → {self.df.index.max().date()}")

    # def _load(self) -> pd.DataFrame:
    #     pattern = os.path.join(DATA_ROOT, "energy_prices", "*.parquet")
    #     files   = sorted(glob.glob(pattern))
    #     if not files:
    #         raise FileNotFoundError("Run p0_download_data.py and p2_build_dataset.py first.")

    #     # Load all three folders
    #     def load_folder(subfolder):
    #         fs = sorted(glob.glob(os.path.join(DATA_ROOT, subfolder, "*.parquet")))
    #         parts = [pd.read_parquet(f) for f in fs]
    #         df = pd.concat(parts, ignore_index=True)
    #         if TIMESTAMP_COL != "__index__":
    #             df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    #             df = df.set_index(TIMESTAMP_COL)
    #         else:
    #             df.index = pd.to_datetime(df.index)
    #         return df.sort_index()
    def _load(self) -> pd.DataFrame:
        pattern = os.path.join(DATA_ROOT, "energy_prices", "*.parquet")
        files   = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError("Run p0_download_data.py and p2_build_dataset.py first.")
        def load_folder(subfolder):
            fs = sorted(glob.glob(os.path.join(DATA_ROOT, subfolder, "*.parquet")))
            parts = [pd.read_parquet(f) for f in fs]
            df = pd.concat(parts)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df.sort_index()

        energy  = load_folder("energy_prices")
        as_pr   = load_folder("as_prices")
        syscond = load_folder("system_conditions")

        df = energy.join(as_pr,   how="outer", rsuffix="_as")
        df = df.join(syscond,     how="outer", rsuffix="_sys")
        
        # Drop unused columns before dropna
        cols_to_drop = [c for c in df.columns if
                        c.startswith("rt_mcpc_") or
                        c.startswith("is_post_rtcb")]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        df = df.ffill(limit=3).dropna()
        # Date split
        if self.split == "train":
            df = df[(df.index >= pd.Timestamp(STAGE1_START)) & (df.index < pd.Timestamp(VAL_START))]
        else:
            df = df[(df.index >= pd.Timestamp(VAL_START)) & (df.index <= pd.Timestamp(STAGE1_END))]

        return df

    def _normalise_price(self, raw: np.ndarray) -> np.ndarray:
        """z-score + ±CLIP_SIGMA clip for price features."""
        mean_p = self.mean[:PRICE_DIM]
        std_p  = self.std[:PRICE_DIM]
        return np.clip((raw - mean_p) / std_p, -CLIP_SIGMA, CLIP_SIGMA)

    def _normalise_system(self, raw: np.ndarray) -> np.ndarray:
        """z-score for system condition features."""
        mean_s = self.mean[PRICE_DIM:]
        std_s  = self.std[PRICE_DIM:]
        return (raw - mean_s) / std_s

    def get_price_window(self, idx: int) -> np.ndarray:
        """Returns (WINDOW_LEN, PRICE_DIM) normalised price window ending at idx."""
        start  = max(0, idx - WINDOW_LEN + 1)
        window = self.df[PRICE_COLS].iloc[start:idx + 1].values.astype(np.float32)
        if len(window) < WINDOW_LEN:
            pad    = np.repeat(window[[0]], WINDOW_LEN - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        return self._normalise_price(window)

    def get_system_vars(self, idx: int) -> np.ndarray:
        """Returns (SYSTEM_DIM,) normalised system vars at idx."""
        raw = self.df[SYSTEM_COLS].iloc[idx].values.astype(np.float32)
        return self._normalise_system(raw)

    @staticmethod
    def time_features(ts: pd.Timestamp) -> np.ndarray:
        """6-dim cyclical encoding: hour-of-day (×2) + day-of-week (×4)."""
        h  = ts.hour + ts.minute / 60
        dw = ts.dayofweek
        return np.array([
            math.sin(2 * math.pi * h  / 24),
            math.cos(2 * math.pi * h  / 24),
            math.sin(2 * math.pi * dw / 7),
            math.cos(2 * math.pi * dw / 7),
            math.sin(4 * math.pi * dw / 7),
            math.cos(4 * math.pi * dw / 7),
        ], dtype=np.float32)

    def get_rt_lmp(self, idx: int) -> float:
        return float(self.df[PRICE_COLS[0]].iloc[idx])

    def get_timestamp(self, idx: int) -> pd.Timestamp:
        return self.df.index[idx]

    def __len__(self):
        return self.n


# ════════════════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════════════════

OBS_FLAT_DIM = WINDOW_LEN * PRICE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM


class ERCOTEnv:
    """
    Steps through the dataset one 5-min interval at a time.
    State  : (price_window, system_vars, time_feats, soc)
    Action : charge/discharge rate ∈ [-1, 1]
    Reward : energy arbitrage revenue for the interval ($)
    """

    def __init__(self, dataset: ERCOTDataset):
        self.ds  = dataset
        self.idx = WINDOW_LEN
        self.soc = 0.5

    def reset(self):
        self.idx = WINDOW_LEN
        self.soc = 0.5
        return self._obs()

    def _obs(self):
        pw  = self.ds.get_price_window(self.idx)
        sv  = self.ds.get_system_vars(self.idx)
        tf  = ERCOTDataset.time_features(self.ds.get_timestamp(self.idx))
        soc = np.array([self.soc], dtype=np.float32)
        return pw, sv, tf, soc

    def step(self, action: float, new_soc: float):
        rt_lmp     = self.ds.get_rt_lmp(self.idx)
        power_mw   = -action * BATTERY_POWER_MW    # +ve when discharging (selling)
        energy_mwh = power_mw * INTERVAL_H
        reward     = energy_mwh * rt_lmp            # $/interval

        self.soc = new_soc
        self.idx += 1
        done = (self.idx >= len(self.ds) - 1)
        return self._obs(), reward, done


# ════════════════════════════════════════════════════════
# REPLAY BUFFER
# ════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self):
        self.buf = deque(maxlen=REPLAY_SIZE)

    def push(self, obs_flat, action, reward, nobs_flat, done):
        self.buf.append((obs_flat, float(action), float(reward), nobs_flat, float(done)))

    def sample(self) -> Tuple:
        batch     = random.sample(self.buf, BATCH_SIZE)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)).to(DEVICE),
            torch.FloatTensor(act).unsqueeze(-1).to(DEVICE),
            torch.FloatTensor(rew).unsqueeze(-1).to(DEVICE),
            torch.FloatTensor(np.array(nobs)).to(DEVICE),
            torch.FloatTensor(done).unsqueeze(-1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ════════════════════════════════════════════════════════
# SAC AGENT
# ════════════════════════════════════════════════════════

def flatten_obs(pw, sv, tf, soc) -> np.ndarray:
    return np.concatenate([pw.flatten(), sv, tf, soc])


def unflatten_obs(flat: torch.Tensor) -> Tuple:
    """Split flat buffer obs → (price_window, system_vars, time_feats, soc) tensors."""
    B      = flat.shape[0]
    pw_dim = WINDOW_LEN * PRICE_DIM
    splits = torch.split(flat, [pw_dim, SYSTEM_DIM, TIME_DIM, SOC_DIM], dim=1)
    pw     = splits[0].view(B, WINDOW_LEN, PRICE_DIM)
    sv     = splits[1]
    tf     = splits[2]
    soc    = splits[3]
    return pw, sv, tf, soc


class SACAgent:
    def __init__(self):
        self.ttfe       = TTFE().to(DEVICE)
        self.actor      = Actor().to(DEVICE)
        self.critic     = Critic().to(DEVICE)
        self.critic_tgt = Critic().to(DEVICE)
        self.proj       = FeasibilityProjection().to(DEVICE)

        # Hard-copy weights to target critic
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

        # TTFE + actor share one optimiser (end-to-end)
        self.opt_actor  = Adam(
            list(self.ttfe.parameters()) + list(self.actor.parameters()),
            lr=LR_ACTOR
        )
        self.opt_critic = Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Learnable entropy coefficient α
        self.log_alpha  = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.opt_alpha  = Adam([self.log_alpha], lr=LR_ALPHA)
        self.tgt_ent    = torch.tensor(TARGET_ENTROPY, device=DEVICE)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def encode(self, pw: torch.Tensor, sv: torch.Tensor,
               tf: torch.Tensor, soc: torch.Tensor) -> torch.Tensor:
        feat = self.ttfe(pw)
        return torch.cat([feat, sv, tf, soc], dim=-1)

    def select_action(self, pw, sv, tf, soc_val: float,
                      deterministic: bool = False) -> Tuple[float, float]:
        pw_t  = torch.FloatTensor(pw).unsqueeze(0).to(DEVICE)
        sv_t  = torch.FloatTensor(sv).unsqueeze(0).to(DEVICE)
        tf_t  = torch.FloatTensor(tf).unsqueeze(0).to(DEVICE)
        soc_t = torch.FloatTensor([[soc_val]]).to(DEVICE)

        with torch.no_grad():
            obs = self.encode(pw_t, sv_t, tf_t, soc_t)
            if deterministic:
                raw = self.actor.get_deterministic_action(obs)
            else:
                raw, _ = self.actor.sample(obs)
            feasible, new_soc = self.proj(raw, soc_t)
        return feasible.item(), new_soc.item()

    def update(self, buffer: ReplayBuffer) -> dict:
        if len(buffer) < BATCH_SIZE:
            return {}

        obs_flat, act, rew, nobs_flat, done = buffer.sample()

        # Unflatten and encode observations
        pw,  sv,  tf,  soc  = unflatten_obs(obs_flat)
        npw, nsv, ntf, nsoc = unflatten_obs(nobs_flat)
        obs_enc  = self.encode(pw,  sv,  tf,  soc)
        nobs_enc = self.encode(npw, nsv, ntf, nsoc)

        # ── Critic update ────────────────────────────────────────────
        with torch.no_grad():
            next_act, next_lp = self.actor.sample(nobs_enc)
            q_tgt = self.critic_tgt.q_min(nobs_enc, next_act)
            y     = rew + GAMMA * (1 - done) * (q_tgt - self.alpha * next_lp)

        q1, q2      = self.critic(obs_enc, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # ── Actor + TTFE update ──────────────────────────────────────
        # Re-encode (detach critic from this pass)
        obs_enc2 = self.encode(pw, sv, tf, soc)
        new_act, log_pi = self.actor.sample(obs_enc2)
        q_new = self.critic.q_min(obs_enc2, new_act)
        actor_loss = (self.alpha.detach() * log_pi - q_new).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # ── Alpha update ─────────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_pi + self.tgt_ent).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # ── Soft target update ────────────────────────────────────────
        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha.item(),
        }

    def save(self, step: int, tag: str = ""):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        fname = f"stage1_{tag or f'step{step}'}.pt"
        torch.save({
            "step":       step,
            "ttfe":       self.ttfe.state_dict(),
            "actor":      self.actor.state_dict(),
            "critic":     self.critic.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "log_alpha":  self.log_alpha.data,
        }, os.path.join(CHECKPOINT_DIR, fname))
        print(f"  [Saved] {fname}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.ttfe.load_state_dict(ckpt["ttfe"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_tgt.load_state_dict(ckpt["critic_tgt"])
        self.log_alpha.data = ckpt["log_alpha"]
        print(f"  [Loaded] step={ckpt['step']} from {path}")
        return ckpt["step"]


# ════════════════════════════════════════════════════════
# QUICK VALIDATION (deterministic rollout on val split)
# ════════════════════════════════════════════════════════

def quick_val(agent: SACAgent, val_dataset: ERCOTDataset,
              max_steps: int = 2000) -> float:
    """Returns total revenue over first max_steps of val split."""
    env   = ERCOTEnv(val_dataset)
    obs   = env.reset()
    pw, sv, tf, soc_arr = obs
    soc_val = float(soc_arr[0])
    total_rev = 0.0

    for _ in range(max_steps):
        action, new_soc = agent.select_action(pw, sv, tf, soc_val, deterministic=True)
        (pw, sv, tf, soc_arr), reward, done = env.step(action, new_soc)
        soc_val   = float(soc_arr[0])
        total_rev += reward
        if done:
            break

    return total_rev


# ════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Pipeline 4 — Stage 1 SAC Training")
    print(f"  Device      : {DEVICE}")
    print(f"  Total steps : {TOTAL_STEPS:,}")
    print(f"  Warmup steps: {WARMUP_STEPS:,}")
    print("=" * 60)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Data
    train_ds = ERCOTDataset("train")
    val_ds   = ERCOTDataset("val")
    train_env = ERCOTEnv(train_ds)

    # Agent + buffer
    agent  = SACAgent()
    buffer = ReplayBuffer()

    # Log file
    log_path = os.path.join(LOG_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["step", "critic_loss", "actor_loss", "alpha",
                     "episode_reward", "val_revenue"])

    # State
    obs_parts  = train_env.reset()
    pw, sv, tf, soc_arr = obs_parts
    soc_val    = float(soc_arr[0])
    ep_reward  = 0.0
    ep_step    = 0
    best_val   = -float("inf")
    recent_losses = {"critic_loss": [], "actor_loss": [], "alpha": []}

    print(f"\nStarting training loop...\n")

    for step in range(1, TOTAL_STEPS + 1):

        # ── Select action ──────────────────────────────────────────
        if step <= WARMUP_STEPS:
            # Random action during warmup
            raw_action = np.random.uniform(-1, 1)
            with torch.no_grad():
                a_t = torch.FloatTensor([[raw_action]]).to(DEVICE)
                s_t = torch.FloatTensor([[soc_val]]).to(DEVICE)
                _, ns_t = agent.proj(a_t, s_t)
            action, new_soc = raw_action, ns_t.item()
        else:
            action, new_soc = agent.select_action(pw, sv, tf, soc_val)

        # ── Environment step ───────────────────────────────────────
        next_obs, reward, done = train_env.step(action, new_soc)
        npw, nsv, ntf, nsoc_arr = next_obs

        # ── Store transition ───────────────────────────────────────
        obs_flat  = flatten_obs(pw,  sv,  tf,  soc_arr)
        nobs_flat = flatten_obs(npw, nsv, ntf, nsoc_arr)
        buffer.push(obs_flat, action, reward, nobs_flat, float(done))

        ep_reward += reward
        ep_step   += 1

        # ── SAC update ─────────────────────────────────────────────
        if step > WARMUP_STEPS:
            info = agent.update(buffer)
            for k, v in info.items():
                recent_losses[k].append(v)

        # ── Advance state ──────────────────────────────────────────
        pw, sv, tf, soc_arr = next_obs
        soc_val = float(nsoc_arr[0])

        # ── Episode reset ──────────────────────────────────────────
        if done:
            obs_parts = train_env.reset()
            pw, sv, tf, soc_arr = obs_parts
            soc_val   = float(soc_arr[0])
            ep_reward = 0.0
            ep_step   = 0

        # ── Logging ────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            cl  = np.mean(recent_losses["critic_loss"][-100:]) if recent_losses["critic_loss"] else 0
            al  = np.mean(recent_losses["actor_loss"][-100:])  if recent_losses["actor_loss"]  else 0
            alp = np.mean(recent_losses["alpha"][-100:])       if recent_losses["alpha"]        else 0
            print(f"  step={step:>7,} | critic={cl:.4f} | actor={al:.4f} | alpha={alp:.4f}")

        # ── Validation ─────────────────────────────────────────────
        if step % EVAL_EVERY == 0 and step > WARMUP_STEPS:
            val_rev = quick_val(agent, val_ds)
            print(f"  ★ Val revenue (2000 steps): ${val_rev:+,.2f}")
            writer.writerow([step, cl, al, alp, ep_reward, val_rev])
            log_file.flush()

            if val_rev > best_val:
                best_val = val_rev
                agent.save(step, tag="best")
                print(f"  ↑ New best: ${best_val:+,.2f}")

        # ── Checkpoint ─────────────────────────────────────────────
        if step % SAVE_EVERY == 0:
            agent.save(step)

    # Final save
    agent.save(TOTAL_STEPS, tag="final")
    log_file.close()

    print("\n" + "=" * 60)
    print("✓ Training complete.")
    print(f"  Best val revenue : ${best_val:+,.2f}")
    print(f"  Logs saved       : {log_path}")
    print("Next step:  python pipeline/p5_evaluate.py")


if __name__ == "__main__":
    main()
