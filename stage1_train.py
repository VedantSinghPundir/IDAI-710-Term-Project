"""
Stage 1: Pre-RTC+B Energy-Only Training
========================================
Two-Stage Deep RL for Battery Storage Bidding (FOML Project)
Author: [Your Name]

Structure
---------
1.  ERCOTDataset          – loads & merges the three parquet folders,
                            builds rolling windows, normalises
2.  ReplayBuffer          – standard off-policy ring buffer
3.  TTFE                  – Transformer Temporal Feature Extractor
                            (32 timesteps × 12 price dims → 64-d vector)
4.  Actor / Critic        – SAC networks (78-d obs → continuous action)
5.  FeasibilityProjection – differentiable SoC / rate clamp
6.  SACAgent              – soft update, entropy tuning, training step
7.  train_stage1()        – main loop with logging & checkpointing

Usage
-----
    python stage1_train.py

Set DATA_ROOT to wherever you cloned the repo's data/processed/ folder.
"""

import os
import glob
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
from typing import Tuple

# ──────────────────────────────────────────────
# CONFIG  (edit these paths / hypers as needed)
# ──────────────────────────────────────────────
DATA_ROOT       = "./data/processed"          # path to cloned repo folder
STAGE1_END_DATE = "2025-12-04"               # last pre-RTC+B date (inclusive)
CHECKPOINT_DIR  = "./checkpoints/stage1"

# Data / window
PRICE_COLS      = None   # auto-detected from energy_prices + as_prices parquets
SYSTEM_COLS     = None   # auto-detected from system_conditions parquets
WINDOW_LEN      = 32     # timesteps fed to TTFE
PRICE_DIM       = 12     # RT LMP + 5 RT MCPC + DAM SPP + 5 DAM AS
SYSTEM_DIM      = 7      # load forecast, actual load, wind, solar, 3 ERCOT indicators
TIME_DIM        = 6      # sin/cos for hour-of-day (2) + day-of-week (4)
SOC_DIM         = 1
OBS_DIM         = 64 + SYSTEM_DIM + TIME_DIM + SOC_DIM   # = 78

# Battery physical limits
SOC_MIN, SOC_MAX   = 0.05, 0.95   # fraction of capacity
POWER_MIN          = -1.0          # max discharge rate (normalised to [-1, 1])
POWER_MAX          =  1.0          # max charge rate
BATTERY_CAP_MWH    = 100.0         # MWh  (adjust to your scenario)
BATTERY_POWER_MW   = 25.0          # MW   (4-hour battery)
EFFICIENCY         = 0.92          # round-trip √ applied per half-step
INTERVAL_H         = 5 / 60        # 5-minute intervals in hours

# SAC hypers
REPLAY_SIZE    = 1_000_000
BATCH_SIZE     = 256
LR_ACTOR       = 3e-4
LR_CRITIC      = 3e-4
LR_ALPHA       = 3e-4
GAMMA          = 0.99
TAU            = 0.005             # soft update coefficient
HIDDEN_DIM     = 256
TTFE_DIM       = 64                # TTFE output size
NUM_HEADS      = 4
NUM_LAYERS     = 2
TARGET_ENTROPY = -1.0              # for 1-dim action space
WARMUP_STEPS   = 5_000
TOTAL_STEPS    = 500_000
LOG_EVERY      = 1_000
SAVE_EVERY     = 50_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════

class ERCOTDataset:
    """
    Merges energy_prices, as_prices, and system_conditions parquet files,
    applies normalisation, and serves (window, system_vars, time_feats, soc)
    tuples for the RL environment.

    Expected parquet schema (auto-detected on first load):
      energy_prices/YYYY-MM.parquet  : timestamp | rt_lmp | dam_spp | ...
      as_prices/YYYY-MM.parquet      : timestamp | rt_mcpc_* | dam_as_* | ...
      system_conditions/YYYY-MM.parquet : timestamp | load_fcst | load_act |
                                          wind_gen | solar_gen | ...
    """

    def __init__(self, data_root: str, end_date: str):
        self.data_root = data_root
        self.end_date  = pd.Timestamp(end_date)
        self.df        = self._load_and_merge()
        self._fit_normaliser()
        print(f"[Dataset] Loaded {len(self.df):,} rows | "
              f"columns: {list(self.df.columns)}")

    # ----------------------------------------------------------
    def _load_folder(self, subfolder: str) -> pd.DataFrame:
        """Load all monthly parquets from a subfolder into one DataFrame."""
        pattern = os.path.join(self.data_root, subfolder, "*.parquet")
        files   = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No parquets found at {pattern}")
        parts = [pd.read_parquet(f) for f in files]
        df = pd.concat(parts, ignore_index=True)
        # Ensure a datetime index named 'timestamp'
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
        else:
            # fallback: assume first column is the timestamp
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.iloc[:, 1:].sort_index()
        return df

    def _load_and_merge(self) -> pd.DataFrame:
        energy  = self._load_folder("energy_prices")
        as_pr   = self._load_folder("as_prices")
        syscond = self._load_folder("system_conditions")

        # Outer join on timestamp index, forward-fill small gaps
        df = energy.join(as_pr,   how="outer", rsuffix="_as")
        df = df.join(syscond,     how="outer", rsuffix="_sys")
        df = df.sort_index().ffill().dropna()

        # Clip to pre-RTC+B window
        df = df[df.index <= self.end_date]

        # ── Store column name groups ────────────────────────────────────
        # Price columns: up to PRICE_DIM (12) — rt_lmp, rt_mcpc_*, dam_spp, dam_as_*
        # Adjust these names to match actual parquet columns!
        price_candidates = [c for c in df.columns if any(
            kw in c.lower() for kw in
            ["lmp", "spp", "mcpc", "dam_as", "regup", "regdn",
             "rrs", "ecrs", "nspin", "reg_up", "reg_dn"]
        )]
        self.price_cols = price_candidates[:PRICE_DIM]

        # System columns: load, wind, solar + ERCOT indicators
        sys_candidates = [c for c in df.columns if any(
            kw in c.lower() for kw in
            ["load", "wind", "solar", "pv", "demand", "ercot"]
        ) and c not in self.price_cols]
        self.system_cols = sys_candidates[:SYSTEM_DIM]

        assert len(self.price_cols)  == PRICE_DIM,  \
            f"Expected {PRICE_DIM} price cols, got {self.price_cols}"
        assert len(self.system_cols) == SYSTEM_DIM, \
            f"Expected {SYSTEM_DIM} system cols, got {self.system_cols}"

        return df

    def _fit_normaliser(self):
        """Compute mean/std on training data for z-score normalisation."""
        all_cols = self.price_cols + self.system_cols
        self.mean = self.df[all_cols].mean()
        self.std  = self.df[all_cols].std().replace(0, 1)

    def normalise(self, df_slice: pd.DataFrame) -> np.ndarray:
        all_cols = self.price_cols + self.system_cols
        return ((df_slice[all_cols] - self.mean) / self.std).values.astype(np.float32)

    @staticmethod
    def time_features(ts: pd.Timestamp) -> np.ndarray:
        """6-dim cyclical time encoding (hour × 2, day-of-week × 4)."""
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

    def __len__(self):
        return len(self.df)

    def get_price_window(self, idx: int) -> np.ndarray:
        """Return (WINDOW_LEN, PRICE_DIM) normalised price window ending at idx."""
        start = max(0, idx - WINDOW_LEN + 1)
        window = self.df[self.price_cols].iloc[start:idx + 1]
        # Pad with first row if we're near the start
        if len(window) < WINDOW_LEN:
            pad = pd.concat([window.iloc[[0]]] * (WINDOW_LEN - len(window)) + [window])
            window = pad
        norm = ((window - self.mean[self.price_cols]) /
                self.std[self.price_cols]).values.astype(np.float32)
        return norm  # shape: (WINDOW_LEN, PRICE_DIM)

    def get_system_vars(self, idx: int) -> np.ndarray:
        """Return (SYSTEM_DIM,) normalised system conditions at idx."""
        row = self.df[self.system_cols].iloc[idx]
        return ((row - self.mean[self.system_cols]) /
                self.std[self.system_cols]).values.astype(np.float32)

    def get_rt_lmp(self, idx: int) -> float:
        """Return raw RT LMP at idx (used for reward computation)."""
        return float(self.df[self.price_cols[0]].iloc[idx])

    def get_timestamp(self, idx: int) -> pd.Timestamp:
        return self.df.index[idx]


# ═══════════════════════════════════════════════════════
# 2. REPLAY BUFFER
# ═══════════════════════════════════════════════════════

class ReplayBuffer:
    """Standard FIFO experience replay buffer."""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)).to(DEVICE),
            torch.FloatTensor(np.array(act)).unsqueeze(-1).to(DEVICE),
            torch.FloatTensor(np.array(rew)).unsqueeze(-1).to(DEVICE),
            torch.FloatTensor(np.array(nobs)).to(DEVICE),
            torch.FloatTensor(np.array(done)).unsqueeze(-1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════════════
# 3. TTFE  –  Transformer Temporal Feature Extractor
# ═══════════════════════════════════════════════════════

class TTFE(nn.Module):
    """
    Encodes a (batch, WINDOW_LEN, PRICE_DIM) price history into a
    (batch, TTFE_DIM) feature vector using multi-head self-attention.

    Follows TempDRL [Li et al. 2024]:
      input projection → positional encoding → N×TransformerEncoder → mean pool
    """

    def __init__(
        self,
        price_dim: int  = PRICE_DIM,
        window_len: int = WINDOW_LEN,
        d_model: int    = 64,
        nhead: int      = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Project raw price dim → d_model
        self.input_proj = nn.Linear(price_dim, d_model)

        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.zeros(1, window_len, d_model))
        nn.init.normal_(self.pos_enc, std=0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model      = d_model,
            nhead        = nhead,
            dim_feedforward = d_model * 4,
            dropout      = dropout,
            batch_first  = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, price_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_window: (batch, WINDOW_LEN, PRICE_DIM)
        Returns:
            feat: (batch, d_model)  — mean-pooled transformer output
        """
        x = self.input_proj(price_window)       # (B, L, d_model)
        x = x + self.pos_enc                    # add positional encoding
        x = self.transformer(x)                 # (B, L, d_model)
        x = self.layer_norm(x)
        feat = x.mean(dim=1)                    # mean pool over time → (B, d_model)
        return feat


# ═══════════════════════════════════════════════════════
# 4. ACTOR & CRITIC
# ═══════════════════════════════════════════════════════

LOG_STD_MIN, LOG_STD_MAX = -5, 2

class Actor(nn.Module):
    """
    Stage 1 actor: outputs a single continuous charge/discharge rate in [-1, 1].
    Uses reparameterisation trick (squashed Gaussian) for SAC.
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h       = self.net(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        return mean, std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action ∈ [-1,1], log_prob) using reparameterisation."""
        mean, std = self(obs)
        dist      = torch.distributions.Normal(mean, std)
        x         = dist.rsample()
        action    = torch.tanh(x)
        # log prob with tanh correction
        log_prob  = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class Critic(nn.Module):
    """Twin Q-networks (double Q trick) to reduce overestimation bias."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden,      hidden), nn.ReLU(),
            nn.Linear(hidden,      1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden,      hidden), nn.ReLU(),
            nn.Linear(hidden,      1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        sa  = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ═══════════════════════════════════════════════════════
# 5. FEASIBILITY PROJECTION
# ═══════════════════════════════════════════════════════

class FeasibilityProjection(nn.Module):
    """
    Differentiable projection of the actor's raw action onto the feasible set:
      - action ∈ [-1, 1] → power_rate ∈ [POWER_MIN, POWER_MAX] MW (normalised)
      - new SoC must stay within [SOC_MIN, SOC_MAX]

    Returns the feasible action (still in [-1,1] scale) and updated SoC.
    """

    def __init__(self):
        super().__init__()

    def forward(self, raw_action: torch.Tensor, soc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            raw_action : (batch, 1) ∈ [-1, 1]
            soc        : (batch, 1) current state-of-charge ∈ [0, 1]
        Returns:
            feasible_action : (batch, 1) clipped to SoC limits
            new_soc         : (batch, 1) updated SoC
        """
        # Convert normalised action → fractional SoC change per interval
        # +1 = full charge at BATTERY_POWER_MW, -1 = full discharge
        delta_soc = raw_action * (BATTERY_POWER_MW * INTERVAL_H / BATTERY_CAP_MWH)

        # Apply efficiency: charging costs more, discharging yields less
        delta_soc_eff = torch.where(
            delta_soc > 0,
            delta_soc * math.sqrt(EFFICIENCY),          # charging loss
            delta_soc / math.sqrt(EFFICIENCY),          # discharging loss
        )

        new_soc = (soc + delta_soc_eff).clamp(SOC_MIN, SOC_MAX)

        # Back-compute feasible delta and feasible action
        feasible_delta = new_soc - soc
        feasible_delta_raw = torch.where(
            feasible_delta > 0,
            feasible_delta / math.sqrt(EFFICIENCY),
            feasible_delta * math.sqrt(EFFICIENCY),
        )
        feasible_action = (feasible_delta_raw /
                           (BATTERY_POWER_MW * INTERVAL_H / BATTERY_CAP_MWH)
                           ).clamp(-1.0, 1.0)

        return feasible_action, new_soc


# ═══════════════════════════════════════════════════════
# 6. SAC AGENT
# ═══════════════════════════════════════════════════════

class SACAgent:
    """
    Soft Actor-Critic with:
      - TTFE for temporal price encoding
      - Automatic entropy tuning
      - Twin delayed critics with soft target updates
    """

    def __init__(self):
        # Networks
        self.ttfe         = TTFE().to(DEVICE)
        self.actor        = Actor().to(DEVICE)
        self.critic       = Critic().to(DEVICE)
        self.critic_tgt   = Critic().to(DEVICE)
        self.projection   = FeasibilityProjection().to(DEVICE)

        # Hard-copy critic → target
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

        # Optimisers (TTFE shares lr with actor)
        self.opt_actor  = Adam(
            list(self.ttfe.parameters()) + list(self.actor.parameters()),
            lr=LR_ACTOR
        )
        self.opt_critic = Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Learnable log-alpha for entropy tuning
        self.log_alpha  = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.opt_alpha  = Adam([self.log_alpha], lr=LR_ALPHA)
        self.target_ent = torch.tensor(TARGET_ENTROPY, device=DEVICE)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ----------------------------------------------------------
    def _encode_obs(self, price_window: torch.Tensor,
                    system_vars: torch.Tensor,
                    time_feats: torch.Tensor,
                    soc: torch.Tensor) -> torch.Tensor:
        """Combine TTFE output + system vars + time feats + SoC → 78-d obs."""
        feat = self.ttfe(price_window)           # (B, 64)
        obs  = torch.cat([feat, system_vars, time_feats, soc], dim=-1)
        return obs                               # (B, 78)

    # ----------------------------------------------------------
    def select_action(self, price_window, system_vars, time_feats,
                      soc_val: float, deterministic: bool = False
                      ) -> Tuple[float, float]:
        """
        Args (all numpy):
            price_window : (WINDOW_LEN, PRICE_DIM)
            system_vars  : (SYSTEM_DIM,)
            time_feats   : (TIME_DIM,)
            soc_val      : scalar float
        Returns:
            action (float in [-1,1]), new_soc (float)
        """
        pw  = torch.FloatTensor(price_window).unsqueeze(0).to(DEVICE)
        sv  = torch.FloatTensor(system_vars).unsqueeze(0).to(DEVICE)
        tf  = torch.FloatTensor(time_feats).unsqueeze(0).to(DEVICE)
        soc = torch.FloatTensor([[soc_val]]).to(DEVICE)

        with torch.no_grad():
            obs = self._encode_obs(pw, sv, tf, soc)
            if deterministic:
                raw_action = self.actor.get_deterministic_action(obs)
            else:
                raw_action, _ = self.actor.sample(obs)
            feasible_action, new_soc = self.projection(raw_action, soc)

        return feasible_action.item(), new_soc.item()

    # ----------------------------------------------------------
    def update(self, replay_buffer: ReplayBuffer) -> dict:
        """One gradient step on critic + actor + alpha."""
        if len(replay_buffer) < BATCH_SIZE:
            return {}

        obs, act, rew, nobs, done = replay_buffer.sample(BATCH_SIZE)

        # ── Critic update ────────────────────────────────────────────────
        with torch.no_grad():
            next_act, next_log_pi = self.actor.sample(nobs)
            q_tgt = self.critic_tgt.q_min(nobs, next_act)
            y     = rew + GAMMA * (1 - done) * (q_tgt - self.alpha * next_log_pi)

        q1, q2       = self.critic(obs, act)
        critic_loss  = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # ── Actor update ─────────────────────────────────────────────────
        new_act, log_pi = self.actor.sample(obs)
        q_new           = self.critic.q_min(obs, new_act)
        actor_loss      = (self.alpha.detach() * log_pi - q_new).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # ── Alpha (entropy) update ────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_pi + self.target_ent).detach()).mean()

        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # ── Soft target update ────────────────────────────────────────────
        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha.item(),
        }

    # ----------------------------------------------------------
    def save(self, step: int):
        """Save all components needed for Stage 2 fine-tuning."""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            "step":          step,
            "ttfe":          self.ttfe.state_dict(),
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_tgt":    self.critic_tgt.state_dict(),
            "log_alpha":     self.log_alpha.data,
            "opt_actor":     self.opt_actor.state_dict(),
            "opt_critic":    self.opt_critic.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f"stage1_step{step}.pt"))
        print(f"[Checkpoint] Saved at step {step}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.ttfe.load_state_dict(ckpt["ttfe"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_tgt.load_state_dict(ckpt["critic_tgt"])
        self.log_alpha.data = ckpt["log_alpha"]
        print(f"[Checkpoint] Loaded from {path} (step {ckpt['step']})")


# ═══════════════════════════════════════════════════════
# 7. ENVIRONMENT (lightweight step-based wrapper)
# ═══════════════════════════════════════════════════════

class ERCOTEnv:
    """
    Thin environment wrapper over ERCOTDataset.
    Steps through each 5-minute interval sequentially.

    State  : (price_window, system_vars, time_feats, soc)
    Action : charge/discharge rate ∈ [-1, 1]
    Reward : energy arbitrage revenue for the interval ($/interval)
    """

    def __init__(self, dataset: ERCOTDataset):
        self.ds      = dataset
        self.n       = len(dataset)
        self.idx     = WINDOW_LEN          # start after first full window
        self.soc     = 0.5                 # initial SoC

    def reset(self):
        self.idx = WINDOW_LEN
        self.soc = 0.5
        return self._get_obs()

    def _get_obs(self):
        pw  = self.ds.get_price_window(self.idx)
        sv  = self.ds.get_system_vars(self.idx)
        tf  = ERCOTDataset.time_features(self.ds.get_timestamp(self.idx))
        soc = np.array([self.soc], dtype=np.float32)
        return pw, sv, tf, soc

    def step(self, action: float, new_soc: float):
        """
        Args:
            action  : feasible action (charge/discharge) ∈ [-1, 1]
            new_soc : SoC after applying feasibility projection
        Returns:
            obs, reward, done
        """
        rt_lmp  = self.ds.get_rt_lmp(self.idx)

        # Energy dispatched this interval (MWh, +ve = discharge = selling)
        power_mw  = -action * BATTERY_POWER_MW   # positive when discharging
        energy_mwh = power_mw * INTERVAL_H
        reward     = energy_mwh * rt_lmp          # $/interval

        self.soc = new_soc
        self.idx += 1
        done = (self.idx >= self.n - 1)

        return self._get_obs(), reward, done


# ═══════════════════════════════════════════════════════
# 8. MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════

def obs_to_flat(pw, sv, tf, soc) -> np.ndarray:
    """Flatten all observation parts for the replay buffer."""
    return np.concatenate([pw.flatten(), sv, tf, soc])


def train_stage1():
    print(f"[Stage 1] Training on device: {DEVICE}")

    # Load data
    dataset = ERCOTDataset(DATA_ROOT, STAGE1_END_DATE)
    env     = ERCOTEnv(dataset)
    agent   = SACAgent()
    buffer  = ReplayBuffer(REPLAY_SIZE)

    obs_parts = env.reset()
    pw, sv, tf, soc_arr = obs_parts
    soc_val = float(soc_arr[0])

    episode_reward = 0.0
    episode_step   = 0
    all_rewards    = []
    losses         = {"critic_loss": [], "actor_loss": [], "alpha": []}

    for step in range(1, TOTAL_STEPS + 1):
        # ── Select action ──────────────────────────────────────────────
        if step < WARMUP_STEPS:
            action  = np.random.uniform(-1, 1)
            # Apply projection manually for SoC bookkeeping
            with torch.no_grad():
                a_t   = torch.FloatTensor([[action]]).to(DEVICE)
                s_t   = torch.FloatTensor([[soc_val]]).to(DEVICE)
                _, ns = agent.projection(a_t, s_t)
            new_soc = ns.item()
        else:
            action, new_soc = agent.select_action(pw, sv, tf, soc_val)

        # ── Environment step ───────────────────────────────────────────
        next_obs_parts, reward, done = env.step(action, new_soc)
        npw, nsv, ntf, nsoc_arr      = next_obs_parts
        new_soc_val                  = float(nsoc_arr[0])

        # ── Store in replay buffer ─────────────────────────────────────
        obs_flat  = obs_to_flat(pw,  sv,  tf,  soc_arr)
        nobs_flat = obs_to_flat(npw, nsv, ntf, nsoc_arr)
        buffer.push(obs_flat, action, reward, nobs_flat, float(done))

        # NOTE: the buffer stores *flat* obs for simplicity.
        # The agent.update() method receives flat obs from the buffer.
        # You need to "unflatten" inside update() — see TODO below.

        episode_reward += reward
        episode_step   += 1

        # ── Update agent ───────────────────────────────────────────────
        if step >= WARMUP_STEPS:
            info = agent.update(buffer)
            for k, v in info.items():
                if v:
                    losses[k].append(v)

        # ── Advance state ──────────────────────────────────────────────
        pw, sv, tf, soc_arr = next_obs_parts
        soc_val             = new_soc_val

        # ── Episode reset ──────────────────────────────────────────────
        if done:
            all_rewards.append(episode_reward)
            print(f"[Step {step:>7}] Episode done | "
                  f"reward={episode_reward:+.2f} | "
                  f"steps={episode_step}")
            obs_parts      = env.reset()
            pw, sv, tf, soc_arr = obs_parts
            soc_val        = float(soc_arr[0])
            episode_reward = 0.0
            episode_step   = 0

        # ── Logging ────────────────────────────────────────────────────
        if step % LOG_EVERY == 0 and losses["critic_loss"]:
            print(f"  step={step} | "
                  f"critic={np.mean(losses['critic_loss'][-100:]):.4f} | "
                  f"actor={np.mean(losses['actor_loss'][-100:]):.4f} | "
                  f"alpha={np.mean(losses['alpha'][-100:]):.4f}")

        # ── Checkpoint ─────────────────────────────────────────────────
        if step % SAVE_EVERY == 0:
            agent.save(step)

    # Final save
    agent.save(TOTAL_STEPS)
    print("[Stage 1] Training complete.")


# ═══════════════════════════════════════════════════════
# TODO LIST  (fill these in before running)
# ═══════════════════════════════════════════════════════
#
# TODO-1: Column names
#   Run this to inspect actual column names in your parquets:
#
#       import pandas as pd
#       df = pd.read_parquet("data/processed/energy_prices/2020-01.parquet")
#       print(df.columns.tolist())
#
#   Then update the keyword filters in ERCOTDataset._load_and_merge()
#   (the `price_candidates` and `sys_candidates` lists).
#
# TODO-2: Unflatten obs in SACAgent.update()
#   The replay buffer stores flat numpy obs. Inside update(), split them:
#
#       pw_dim  = WINDOW_LEN * PRICE_DIM   # 32 * 12 = 384
#       sv_dim  = SYSTEM_DIM               # 7
#       tf_dim  = TIME_DIM                 # 6
#       soc_dim = SOC_DIM                  # 1
#
#       pw  = obs[:, :pw_dim].view(B, WINDOW_LEN, PRICE_DIM)
#       sv  = obs[:, pw_dim:pw_dim+sv_dim]
#       tf  = obs[:, pw_dim+sv_dim:pw_dim+sv_dim+tf_dim]
#       soc = obs[:, -soc_dim:]
#       obs_enc = self._encode_obs(pw, sv, tf, soc)
#
#   Replace the `obs` and `nobs` tensor uses in update() with `obs_enc`.
#
# TODO-3: Reward shaping (optional)
#   The current reward is raw arbitrage revenue. You may want to add a
#   battery degradation penalty (see Cao et al. [14] in the paper).
#
# TODO-4: Evaluation callback
#   Add a held-out validation month (e.g., 2025-11) and compute:
#     - Total revenue vs rule-based heuristic
#     - % of Perfect Information Optimisation (PIO) upper bound
#
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    train_stage1()
