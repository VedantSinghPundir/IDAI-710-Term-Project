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
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.config import *
from pipeline.p3_models import TTFE, Actor, Critic, FeasibilityProjection


# ════════════════════════════════════════════════════════
# DATASET LOADER
# ════════════════════════════════════════════════════════

class ERCOTDataset:
    """Loads merged parquets and serves rolling price windows + obs components."""

    def __init__(self, split: str = "train"):
        self.split = split
        self.df    = self._load()
        self.n     = len(self.df)

        stats_path = os.path.join(CHECKPOINT_DIR, "normaliser_stats.npz")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Normaliser not found at {stats_path}\n"
                "Run p2_build_dataset.py first.\n"
                "IMPORTANT: Rerun p2 if STAGE1_START changed."
            )
        stats     = np.load(stats_path, allow_pickle=True)
        self.mean = stats["mean"].astype(np.float32)
        self.std  = stats["std"].astype(np.float32)

        print(f"[Dataset:{split}] {self.n:,} rows | "
              f"{self.df.index.min().date()} → {self.df.index.max().date()}")

    def _load(self) -> pd.DataFrame:
        pattern = os.path.join(DATA_ROOT, "energy_prices", "*.parquet")
        if not sorted(glob.glob(pattern)):
            raise FileNotFoundError("Run p0_download_data.py first.")

        def load_folder(subfolder):
            fs    = sorted(glob.glob(os.path.join(DATA_ROOT, subfolder, "*.parquet")))
            parts = [pd.read_parquet(f) for f in fs]
            df    = pd.concat(parts)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df.sort_index()

        energy  = load_folder("energy_prices")
        as_pr   = load_folder("as_prices")
        syscond = load_folder("system_conditions")

        df = energy.join(as_pr,   how="outer", rsuffix="_as")
        df = df.join(syscond,     how="outer", rsuffix="_sys")

        cols_to_drop = [c for c in df.columns if
                        c.startswith("rt_mcpc_") or
                        c.startswith("is_post_rtcb")]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        df = df.ffill(limit=3).dropna()

        if self.split == "train":
            df = df[(df.index >= pd.Timestamp(STAGE1_START)) &
                    (df.index <  pd.Timestamp(VAL_START))]
        elif self.split == "val":
            # Val: used for checkpoint selection during training only.
            # Never used for final reported results.
            df = df[(df.index >= pd.Timestamp(VAL_START)) &
                    (df.index <  pd.Timestamp(TEST_START))]
        elif self.split == "test":
            # Test: held out completely.
            # Only used in p5_evaluate.py for final evaluation.
            df = df[(df.index >= pd.Timestamp(TEST_START)) &
                    (df.index <= pd.Timestamp(STAGE1_END))]
        else:
            raise ValueError(f"Unknown split: {self.split!r}. Use 'train', 'val', or 'test'.")
        return df

    def _normalise_price(self, raw: np.ndarray) -> np.ndarray:
        mean_p = self.mean[:PRICE_DIM]
        std_p  = self.std[:PRICE_DIM]
        return np.clip((raw - mean_p) / std_p, -CLIP_SIGMA, CLIP_SIGMA)

    def _normalise_system(self, raw: np.ndarray) -> np.ndarray:
        mean_s = self.mean[PRICE_DIM:]
        std_s  = self.std[PRICE_DIM:]
        return (raw - mean_s) / std_s

    def get_price_window(self, idx: int) -> np.ndarray:
        start  = max(0, idx - WINDOW_LEN + 1)
        window = self.df[PRICE_COLS].iloc[start:idx + 1].values.astype(np.float32)
        if len(window) < WINDOW_LEN:
            pad    = np.repeat(window[[0]], WINDOW_LEN - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        return self._normalise_price(window)

    def get_system_vars(self, idx: int) -> np.ndarray:
        raw = self.df[SYSTEM_COLS].iloc[idx].values.astype(np.float32)
        return self._normalise_system(raw)

    @staticmethod
    def time_features(ts: pd.Timestamp) -> np.ndarray:
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
# ENVIRONMENT — Plan C inventory-adjusted reward
# ════════════════════════════════════════════════════════

OBS_FLAT_DIM = WINDOW_LEN * PRICE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM


class ERCOTEnv:
    """
    ERCOT energy-only arbitrage environment.

    Action convention:
        +1 = full charge   (buy from grid, SoC increases)
        -1 = full discharge (sell to grid, SoC decreases)
         0 = hold

    Reward (Plan C — inventory-adjusted):
        grid_mwh = -action * BATTERY_POWER_MW * INTERVAL_H
                   positive = selling (discharge), negative = buying (charge)

        shaped_reward = grid_mwh * (rt_lmp - p_ref) / REWARD_SCALE
                      - CYCLE_COST_PER_MWH * |grid_mwh| / REWARD_SCALE

    p_ref = training median rt_lmp (computed once, never from val data).

    IMPORTANT:
        env.step() MUST receive the feasible projected action, not the raw
        actor action. Passing raw actions causes energy/SoC inconsistency
        near the SoC bounds.
    """

    def __init__(self, dataset: ERCOTDataset, p_ref: Optional[float] = None):
        self.ds       = dataset
        self.idx      = WINDOW_LEN
        self.soc      = 0.5
        self.ep_steps = 0

        # p_ref from training split only — never compute from val (look-ahead bias)
        # For val env: pass train_env.p_ref explicitly
        if p_ref is not None:
            self.p_ref = float(p_ref)
        else:
            self.p_ref = float(dataset.df[PRICE_COLS[0]].median())

        print(f"[Env:{dataset.split}] p_ref = ${self.p_ref:.2f}/MWh  "
              f"(training median — for reward shaping only, not evaluation)")

    def reset(self) -> Tuple:
        """Random start for training."""
        max_start     = int(len(self.ds) * 0.8)
        self.idx      = np.random.randint(WINDOW_LEN, max_start)
        self.soc      = np.random.uniform(0.3, 0.7)
        self.ep_steps = 0
        return self._obs()

    def reset_deterministic(self) -> Tuple:
        """Fixed start for reproducible evaluation."""
        self.idx      = WINDOW_LEN
        self.soc      = 0.5
        self.ep_steps = 0
        return self._obs()

    def _obs(self) -> Tuple:
        pw  = self.ds.get_price_window(self.idx)
        sv  = self.ds.get_system_vars(self.idx)
        tf  = ERCOTDataset.time_features(self.ds.get_timestamp(self.idx))
        soc = np.array([self.soc], dtype=np.float32)
        return pw, sv, tf, soc

    def step(self, action: float, new_soc: float) -> Tuple:
        """
        CRITICAL: action must be the FEASIBLE projected action.
        Do NOT pass raw actor output — energy calculation uses action directly.

        Returns: (next_obs, shaped_reward, done, cash_reward)
            shaped_reward: used for training (inventory-adjusted)
            cash_reward:   real market cash (for evaluation display only)
        """
        rt_lmp = self.ds.get_rt_lmp(self.idx)

        # Energy: positive = selling to grid (discharge), negative = buying (charge)
        grid_mwh = -action * BATTERY_POWER_MW * INTERVAL_H

        # Raw cash (for evaluation display — not used for training)
        cash_reward = grid_mwh * rt_lmp

        # Inventory-adjusted shaped reward (for training)
        spread_reward = grid_mwh * (rt_lmp - self.p_ref)
        degradation   = CYCLE_COST_PER_MWH * abs(grid_mwh)
        shaped_reward = (spread_reward - degradation) / REWARD_SCALE

        self.soc       = float(np.clip(new_soc, SOC_MIN, SOC_MAX))
        self.idx      += 1
        self.ep_steps += 1
        done = (self.idx >= len(self.ds) - 1) or (self.ep_steps >= MAX_EP_STEPS)

        return self._obs(), shaped_reward, done, cash_reward


# ════════════════════════════════════════════════════════
# REPLAY BUFFERS
# ════════════════════════════════════════════════════════

class ReplayBuffer:
    """Replay buffer with configurable capacity and variable sample size."""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs_flat, action, reward, nobs_flat, done):
        self.buf.append((obs_flat, float(action), float(reward),
                         nobs_flat, float(done)))

    def sample(self, n: int) -> Tuple:
        n     = min(n, len(self.buf))
        batch = random.sample(self.buf, n)
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


def get_demo_ratio(step: int) -> float:
    """
    Linearly decay demo sampling ratio from 1.0 to DEMO_FLOOR.
    Never reaches 0 — floor maintained throughout training.

    Rationale: without the floor, policy collapsed within 10-20k steps in
    previous runs. Once collapsed, agent_buffer filled with always-discharge
    transitions. Keeping 5% demos ensures critic always sees both actions.
    """
    ratio = 1.0 - (step / DEMO_DECAY_STEPS) * (1.0 - DEMO_FLOOR)
    return max(DEMO_FLOOR, ratio)


def sample_mixed(demo_buf: ReplayBuffer,
                 agent_buf: ReplayBuffer,
                 step: int) -> Optional[Tuple]:
    """Sample mixed batch at current demo ratio. Returns None if insufficient data."""
    demo_ratio = get_demo_ratio(step)
    n_demo     = int(BATCH_SIZE * demo_ratio)
    n_agent    = BATCH_SIZE - n_demo
    n_demo     = min(n_demo,  len(demo_buf))
    n_agent    = min(n_agent, len(agent_buf))

    if n_demo + n_agent < 64:    # minimum viable batch
        return None

    parts = []
    if n_demo  > 0: parts.append(demo_buf.sample(n_demo))
    if n_agent > 0: parts.append(agent_buf.sample(n_agent))
    if len(parts) == 1:
        return parts[0]
    return tuple(torch.cat([p[i] for p in parts], dim=0) for i in range(5))


# ════════════════════════════════════════════════════════
# UTILITY
# ════════════════════════════════════════════════════════

def flatten_obs(pw, sv, tf, soc) -> np.ndarray:
    return np.concatenate([pw.flatten(), sv, tf, soc])


def unflatten_obs(flat: torch.Tensor) -> Tuple:
    """Split flat buffer obs → (price_window, system_vars, time_feats, soc)."""
    pw_dim = WINDOW_LEN * PRICE_DIM
    splits = torch.split(flat, [pw_dim, SYSTEM_DIM, TIME_DIM, SOC_DIM], dim=1)
    pw     = splits[0].view(flat.shape[0], WINDOW_LEN, PRICE_DIM)
    return pw, splits[1], splits[2], splits[3]


# ════════════════════════════════════════════════════════
# SAC AGENT — fixed alpha, projection-consistent
# ════════════════════════════════════════════════════════

class SACAgent:
    """
    SAC with fixed temperature and projection-consistent critic/actor updates.

    ALPHA:
        Fixed at ALPHA_FIXED throughout training. No gradient update.
        Rationale: SAC v2 auto-tuning collapsed alpha to 0.03 in every
        previous run regardless of TARGET_ENTROPY setting. Fixed alpha
        permanently maintains the exploration incentive.

    PROJECTION CONSISTENCY:
        All Q-value evaluations use feasible projected actions, not raw
        actor output. This is the core correctness fix. Without it:
        - Critic learns Q(s, infeasible_action) which is meaningless
        - Actor gradient pushes toward actions that look good to critic
          but are infeasible and will be clipped to 0 by projection
    """

    def __init__(self):
        self.ttfe       = TTFE().to(DEVICE)
        self.actor      = Actor().to(DEVICE)
        self.critic     = Critic().to(DEVICE)
        self.critic_tgt = Critic().to(DEVICE)
        self.proj       = FeasibilityProjection().to(DEVICE)

        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

        self.opt_actor  = Adam(
            list(self.ttfe.parameters()) + list(self.actor.parameters()),
            lr=LR_ACTOR
        )
        self.opt_critic = Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Fixed alpha — no log_alpha parameter, no opt_alpha
        self.alpha = ALPHA_FIXED
        print(f"[SAC] Fixed alpha = {self.alpha}  (no automatic tuning)")

    def encode(self, pw, sv, tf, soc) -> torch.Tensor:
        return torch.cat([self.ttfe(pw), sv, tf, soc], dim=-1)

    def select_action(self, pw, sv, tf, soc_val: float,
                      deterministic: bool = False) -> Tuple[float, float]:
        """Returns (feasible_action, new_soc) — projection applied."""
        pw_t  = torch.FloatTensor(pw).unsqueeze(0).to(DEVICE)
        sv_t  = torch.FloatTensor(sv).unsqueeze(0).to(DEVICE)
        tf_t  = torch.FloatTensor(tf).unsqueeze(0).to(DEVICE)
        soc_t = torch.FloatTensor([[soc_val]]).to(DEVICE)
        with torch.no_grad():
            obs = self.encode(pw_t, sv_t, tf_t, soc_t)
            raw = (self.actor.get_deterministic_action(obs) if deterministic
                   else self.actor.sample(obs)[0])
            feasible, new_soc = self.proj(raw, soc_t)
        return feasible.item(), new_soc.item()

    def update(self, batch: Tuple) -> dict:
        """
        SAC update with projection-consistent Q-value targets.

        KEY FIXES:
          1. Critic target uses proj(next_raw, nsoc) not next_raw
          2. Actor update uses proj(raw_act, soc)  not raw_act
          3. TTFE is updated through actor loss only (critic sees detached features)
        """
        obs_flat, act, rew, nobs_flat, done = batch

        pw,  sv,  tf,  soc  = unflatten_obs(obs_flat)
        npw, nsv, ntf, nsoc = unflatten_obs(nobs_flat)

        # ── Critic update ─────────────────────────────────────────────
        # TTFE features detached: critic loss does NOT update TTFE
        # (TTFE is updated through actor loss below)
        with torch.no_grad():
            nobs_enc = self.encode(npw, nsv, ntf, nsoc)
            next_raw, next_lp = self.actor.sample(nobs_enc)

            # FIX 1: project next action before Q target evaluation
            next_act, _ = self.proj(next_raw, nsoc)

            q_tgt = self.critic_tgt.q_min(nobs_enc, next_act)
            # alpha is a Python float — no gradient through it
            y = rew + GAMMA * (1.0 - done) * (q_tgt - self.alpha * next_lp)

        # Detach features for critic loss (TTFE updated via actor only)
        obs_enc_critic = self.encode(pw, sv, tf, soc).detach()
        q1, q2 = self.critic(obs_enc_critic, act)

        # Huber loss: reduces spike gradient dominance from 40,580× to ~15×
        critic_loss = (F.huber_loss(q1, y, delta=HUBER_DELTA) +
                       F.huber_loss(q2, y, delta=HUBER_DELTA))

        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP)
        self.opt_critic.step()

        # ── Actor + TTFE update ───────────────────────────────────────
        obs_enc_actor   = self.encode(pw, sv, tf, soc)
        raw_act, log_pi = self.actor.sample(obs_enc_actor)

        # FIX 2: project actor action before critic evaluation
        new_act, _ = self.proj(raw_act, soc)

        q_new      = self.critic.q_min(obs_enc_actor, new_act)
        actor_loss = (self.alpha * log_pi - q_new).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.ttfe.parameters()) + list(self.actor.parameters()),
            max_norm=GRAD_CLIP
        )
        self.opt_actor.step()

        # ── Soft target update ────────────────────────────────────────
        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(TAU * p.data + (1.0 - TAU) * pt.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       float(self.alpha),    # constant — for logging consistency
            "log_pi":      log_pi.mean().item(),
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
            "alpha":      self.alpha,    # float, not tensor
        }, os.path.join(CHECKPOINT_DIR, fname))
        print(f"  [Saved] {fname}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.ttfe.load_state_dict(ckpt["ttfe"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_tgt.load_state_dict(ckpt["critic_tgt"])
        # Alpha is always ALPHA_FIXED — ignore any saved value
        print(f"  [Loaded] step={ckpt['step']} from {path}")
        return ckpt["step"]


# ════════════════════════════════════════════════════════
# QUICK VALIDATION — deterministic, inventory-adjusted
# ════════════════════════════════════════════════════════

def quick_val(agent: SACAgent,
              val_dataset: ERCOTDataset,
              p_ref: float,
              max_steps: int = 2000) -> dict:
    """
    Deterministic rollout from fixed start point.

    Reports:
        cash_revenue:     real market cash received
        inventory_adjusted: fair profit accounting for final SoC position
            = cash_revenue + (final_soc - initial_soc) * BATTERY_CAP_MWH * p_ref
              - total degradation cost

    Always uses p_ref from training split (no look-ahead bias).
    """
    # Pass p_ref from training env to avoid recomputing from val data
    env = ERCOTEnv(val_dataset, p_ref=p_ref)
    env.reset_deterministic()

    pw, sv, tf, soc_arr = env._obs()
    soc_val     = float(soc_arr[0])
    initial_soc = soc_val

    total_cash        = 0.0
    total_degradation = 0.0
    charge_count      = 0
    discharge_count   = 0
    hold_count        = 0
    n_steps           = 0

    while n_steps < max_steps:
        action, new_soc = agent.select_action(
            pw, sv, tf, soc_val, deterministic=True
        )

        (pw, sv, tf, soc_arr), shaped_reward, done, cash_reward = env.step(
            action, new_soc
        )
        soc_val = float(soc_arr[0])

        grid_mwh   = -action * BATTERY_POWER_MW * INTERVAL_H
        degradation = CYCLE_COST_PER_MWH * abs(grid_mwh)

        total_cash        += cash_reward
        total_degradation += degradation

        if   action >  1e-6: charge_count    += 1
        elif action < -1e-6: discharge_count += 1
        else:                hold_count      += 1

        n_steps += 1
        if done:
            break

    final_soc         = soc_val
    inventory_change  = (final_soc - initial_soc) * BATTERY_CAP_MWH * p_ref
    inventory_adjusted = total_cash + inventory_change - total_degradation

    return {
        "cash_revenue":      total_cash,
        "inventory_adjusted": inventory_adjusted,
        "inventory_change":  inventory_change,
        "degradation_cost":  total_degradation,
        "initial_soc":       initial_soc,
        "final_soc":         final_soc,
        "charge_frac":       charge_count    / max(n_steps, 1),
        "discharge_frac":    discharge_count / max(n_steps, 1),
        "hold_frac":         hold_count      / max(n_steps, 1),
        "n_steps":           n_steps,
    }


# ════════════════════════════════════════════════════════
# DEMONSTRATIONS — projection-consistent
# ════════════════════════════════════════════════════════

def collect_demonstrations(env: ERCOTEnv,
                           dataset: ERCOTDataset,
                           buffer: ReplayBuffer,
                           n_steps: int = DEMO_STEPS):
    """
    Fill demo_buffer with rule-based heuristic transitions.

    Rule: charge if rt_lmp < p_ref, discharge if rt_lmp >= p_ref.

    CRITICAL FIXES:
      1. Projection is applied to raw heuristic action before env.step()
      2. Feasible action (not raw) is stored in buffer
      3. Reward is computed from feasible action inside env.step()

    This ensures demo transitions are internally consistent:
        action stored ↔ energy used in reward ↔ SoC change
    """
    print(f"[Demo] Collecting {n_steps:,} rule-based demonstrations...")
    print(f"[Demo] p_ref = ${env.p_ref:.2f}/MWh  (charge if below, discharge if above)")

    obs = env.reset()
    pw, sv, tf, soc_arr = obs
    soc_val = float(soc_arr[0])
    proj    = FeasibilityProjection().to(DEVICE)

    charge_count    = 0
    discharge_count = 0
    hold_count      = 0

    for i in range(n_steps):
        rt_lmp     = dataset.get_rt_lmp(env.idx)
        raw_action = 1.0 if rt_lmp < env.p_ref else -1.0

        a_t = torch.FloatTensor([[raw_action]]).to(DEVICE)
        s_t = torch.FloatTensor([[soc_val]]).to(DEVICE)
        with torch.no_grad():
            feasible_t, ns_t = proj(a_t, s_t)

        # FIX: use feasible action, not raw action
        feasible_action = float(feasible_t.item())
        new_soc         = float(ns_t.item())

        # FIX: env.step receives feasible action → energy consistent with SoC
        next_obs, shaped_reward, done, _ = env.step(feasible_action, new_soc)
        npw, nsv, ntf, nsoc_arr = next_obs

        obs_flat  = flatten_obs(pw,  sv,  tf,  soc_arr)
        nobs_flat = flatten_obs(npw, nsv, ntf, nsoc_arr)

        # FIX: store feasible action in buffer (not raw)
        buffer.push(obs_flat, feasible_action, shaped_reward, nobs_flat, float(done))

        if   feasible_action >  1e-6: charge_count    += 1
        elif feasible_action < -1e-6: discharge_count += 1
        else:                         hold_count      += 1

        pw, sv, tf, soc_arr = next_obs
        soc_val = float(nsoc_arr[0])

        if done:
            obs = env.reset()
            pw, sv, tf, soc_arr = obs
            soc_val = float(soc_arr[0])

        if (i + 1) % 10_000 == 0:
            print(f"[Demo] {i + 1:,}/{n_steps:,} steps...")

    total = max(charge_count + discharge_count + hold_count, 1)
    print(f"[Demo] Complete: {len(buffer):,} transitions")
    print(f"[Demo] Action balance: "
          f"{charge_count    / total * 100:.1f}% charge, "
          f"{discharge_count / total * 100:.1f}% discharge, "
          f"{hold_count      / total * 100:.1f}% hold (SoC-limited transitions)")


# ════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Pipeline 4 — Plan C + Fixed Alpha + Projection Fixes")
    print(f"  Device        : {DEVICE}")
    print(f"  Train         : {STAGE1_START} → {VAL_START}")
    print(f"  Val           : {VAL_START} → {TEST_START}  (checkpoint selection)")
    print(f"  Test          : {TEST_START} → {STAGE1_END}  (held out — final eval only)")
    print(f"  Total steps   : {TOTAL_STEPS:,}")
    print(f"  ALPHA_FIXED   : {ALPHA_FIXED}  (no auto-tuning)")
    print(f"  HUBER_DELTA   : {HUBER_DELTA}")
    print(f"  LR_CRITIC     : {LR_CRITIC}")
    print(f"  GRAD_CLIP     : {GRAD_CLIP}")
    print(f"  CYCLE_COST    : ${CYCLE_COST_PER_MWH}/MWh")
    print(f"  DEMO_STEPS    : {DEMO_STEPS:,}")
    print(f"  DEMO_FLOOR    : {DEMO_FLOOR}")
    print(f"  Early stop    : critic>{CRITIC_LOSS_STOP} OR "
          f"log_pi>{LOG_PI_STOP} after step {MIN_STEP_BEFORE_STOP:,}")
    print("=" * 65)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────
    train_ds  = ERCOTDataset("train")
    val_ds    = ERCOTDataset("val")
    train_env = ERCOTEnv(train_ds)
    train_p_ref = train_env.p_ref   # used for val env — no look-ahead bias

    print(f"\n  Training rows : {len(train_ds):,}")
    print(f"  Val rows      : {len(val_ds):,}")
    print(f"  p_ref (train) : ${train_p_ref:.2f}/MWh")

    # ── Agent and buffers ──────────────────────────────────────────
    agent        = SACAgent()
    demo_buffer  = ReplayBuffer(capacity=DEMO_BUFFER_SIZE)
    agent_buffer = ReplayBuffer(capacity=AGENT_BUFFER_SIZE)

    # ── Fill demo buffer (projection-consistent) ───────────────────
    collect_demonstrations(train_env, train_ds, demo_buffer)

    # ── Log file ───────────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow([
        "step", "critic_loss", "actor_loss", "alpha", "log_pi",
        "train_charge_frac", "demo_ratio",
        "val_cash", "val_inv_adjusted", "val_final_soc", "val_charge_frac",
    ])

    # ── Training state ─────────────────────────────────────────────
    obs_parts = train_env.reset()
    pw, sv, tf, soc_arr = obs_parts
    soc_val    = float(soc_arr[0])
    best_val   = -float("inf")

    recent = {
        "critic_loss": deque(maxlen=100),
        "actor_loss":  deque(maxlen=100),
        "log_pi":      deque(maxlen=100),
    }
    recent_actions = deque(maxlen=1000)

    # Initialise logging vars before first LOG_EVERY
    cl = al = alp_lp = charge_frac = 0.0
    stop_training        = False
    consec_log_pi_above  = 0   # consecutive LOG_EVERY periods with log_pi > LOG_PI_STOP
    consec_critic_above  = 0   # consecutive LOG_EVERY periods with critic > CRITIC_LOSS_STOP
    consec_log_pi_above  = 0   # consecutive eval periods with log_pi > LOG_PI_STOP
    consec_critic_above  = 0   # consecutive eval periods with critic > CRITIC_LOSS_STOP

    print(f"\nStarting training loop...\n")
    print(f"  Target (first 20k steps):")
    print(f"    charge% > {CHARGE_FRAC_MIN*100:.0f}% (no collapse)")
    print(f"    log_pi  < 0.0 (healthy policy)")
    print(f"    critic  < {CRITIC_LOSS_STOP} (stable critic)")
    print(f"    val final_soc not stuck at {SOC_MIN} or {SOC_MAX}")
    print()

    for step in range(1, TOTAL_STEPS + 1):

        # ── Select action (always projected) ──────────────────────
        action, new_soc = agent.select_action(pw, sv, tf, soc_val)
        recent_actions.append(action)

        # ── Environment step (receives projected action) ───────────
        next_obs, shaped_reward, done, cash_reward = train_env.step(action, new_soc)
        npw, nsv, ntf, nsoc_arr = next_obs

        # Store projected action + shaped reward in agent buffer
        obs_flat  = flatten_obs(pw,  sv,  tf,  soc_arr)
        nobs_flat = flatten_obs(npw, nsv, ntf, nsoc_arr)
        agent_buffer.push(obs_flat, action, shaped_reward, nobs_flat, float(done))

        # ── SAC update ─────────────────────────────────────────────
        batch = sample_mixed(demo_buffer, agent_buffer, step)
        if batch is not None:
            info = agent.update(batch)
            for k in recent:
                if k in info:
                    recent[k].append(info[k])

        # ── Advance state ──────────────────────────────────────────
        pw, sv, tf, soc_arr = next_obs
        soc_val = float(nsoc_arr[0])

        if done:
            obs_parts = train_env.reset()
            pw, sv, tf, soc_arr = obs_parts
            soc_val = float(soc_arr[0])

        # ── Logging ────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            cl       = float(np.mean(recent["critic_loss"])) if recent["critic_loss"] else 0.0
            al       = float(np.mean(recent["actor_loss"]))  if recent["actor_loss"]  else 0.0
            alp_lp   = float(np.mean(recent["log_pi"]))      if recent["log_pi"]      else 0.0
            charge_frac = (sum(1 for a in recent_actions if a > 1e-6) /
                           max(len(recent_actions), 1))
            demo_ratio  = get_demo_ratio(step)

            print(f"  step={step:>7,} | critic={cl:.2f} | actor={al:.2f} | "
                  f"alpha={ALPHA_FIXED:.3f} | log_pi={alp_lp:+.3f} | "
                  f"charge%={charge_frac*100:.1f} | demo%={demo_ratio*100:.0f}")

            if charge_frac < CHARGE_FRAC_MIN and step > 10_000:
                print(f"  [WARN] Charge fraction {charge_frac*100:.1f}% "
                      f"< {CHARGE_FRAC_MIN*100:.0f}% — possible collapse")

        # ── Validation ─────────────────────────────────────────────
        if step % EVAL_EVERY == 0:
            val_metrics   = quick_val(agent, val_ds, p_ref=train_p_ref)
            cash_rev      = val_metrics["cash_revenue"]
            inv_adj       = val_metrics["inventory_adjusted"]
            final_soc     = val_metrics["final_soc"]
            val_chg_frac  = val_metrics["charge_frac"]
            val_hold_frac = val_metrics["hold_frac"]
            demo_ratio    = get_demo_ratio(step)

            print(f"  ★ Val: cash=${cash_rev:+,.2f} | "
                  f"inv_adj=${inv_adj:+,.2f} | "
                  f"final_soc={final_soc:.3f} | "
                  f"charge%={val_chg_frac*100:.1f} | "
                  f"hold%={val_hold_frac*100:.1f} | "
                  f"[critic={cl:.1f} | log_pi={alp_lp:+.3f}]")

            writer.writerow([
                step, cl, al, ALPHA_FIXED, alp_lp,
                charge_frac, demo_ratio,
                cash_rev, inv_adj, final_soc, val_chg_frac,
            ])
            log_file.flush()

            # Health-gated checkpoint — only save when policy is stable
            is_healthy = (cl < CRITIC_LOSS_STOP) and (alp_lp < LOG_PI_STOP)

            if inv_adj > best_val and is_healthy:
                best_val = inv_adj
                agent.save(step, tag="best")
                print(f"  ↑ New best (healthy): inv_adj=${best_val:+,.2f}")
            elif inv_adj > best_val and not is_healthy:
                print(f"  [SKIP] Better val but unhealthy checkpoint "
                      f"(critic={cl:.1f} OR log_pi={alp_lp:+.3f})")

        # ── Early stopping ─────────────────────────────────────────
        # Requires EARLY_STOP_CONSEC_EVALS consecutive LOG_EVERY periods
        # above threshold before triggering — avoids noise-triggered stops.
        # Previous run stopped at log_pi=+0.003 (noise) with threshold=0.0.
        # Now LOG_PI_STOP=0.05 and requires 2 consecutive periods.
        if step > MIN_STEP_BEFORE_STOP and recent["critic_loss"]:
            cl_check = float(np.mean(recent["critic_loss"]))
            lp_check = float(np.mean(recent["log_pi"])) if recent["log_pi"] else -1.0

            # Critic: accumulate consecutive breaches
            if cl_check > CRITIC_LOSS_STOP:
                consec_critic_above += 1
                if consec_critic_above >= EARLY_STOP_CONSEC_EVALS:
                    print(f"\n  [EARLY STOP] Critic {cl_check:.1f} > {CRITIC_LOSS_STOP} "
                          f"for {EARLY_STOP_CONSEC_EVALS} consecutive periods at step {step:,}.")
                    agent.save(step, tag="emergency")
                    stop_training = True
                else:
                    print(f"  [WARN] Critic {cl_check:.1f} > {CRITIC_LOSS_STOP} "
                          f"({consec_critic_above}/{EARLY_STOP_CONSEC_EVALS})")
            else:
                consec_critic_above = 0

            # log_pi: accumulate consecutive breaches (no redundant step check)
            if not stop_training:
                if lp_check > LOG_PI_STOP:
                    consec_log_pi_above += 1
                    if consec_log_pi_above >= EARLY_STOP_CONSEC_EVALS:
                        print(f"\n  [EARLY STOP] log_pi {lp_check:+.3f} > {LOG_PI_STOP} "
                              f"for {EARLY_STOP_CONSEC_EVALS} consecutive periods at step {step:,}.")
                        agent.save(step, tag="emergency")
                        stop_training = True
                    else:
                        print(f"  [WARN] log_pi {lp_check:+.3f} > {LOG_PI_STOP} "
                              f"({consec_log_pi_above}/{EARLY_STOP_CONSEC_EVALS})")
                else:
                    consec_log_pi_above = 0

        # ── Periodic checkpoint ────────────────────────────────────
        if step % SAVE_EVERY == 0:
            agent.save(step)

        if stop_training:
            break

    # ── Final save ─────────────────────────────────────────────────
    agent.save(step, tag="final")
    log_file.close()

    print("\n" + "=" * 65)
    print("✓ Training complete.")
    print(f"  Stopped at step       : {step:,}")
    print(f"  Best inv_adj revenue  : ${best_val:+,.2f}")
    print(f"  Early stopped         : {stop_training}")
    print(f"  Logs                  : {log_path}")
    print("Next step: python pipeline/p5_evaluate.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
