# """
# Pipeline 4 — SAC Training Loop
# ================================
# Trains the Stage 1 SAC + TTFE agent on pre-RTC+B ERCOT data.

# Run AFTER p0, p1, p2 (data ready), p3 (model shapes verified).

# Usage:
#     python pipeline/p4_train.py

# Outputs:
#     checkpoints/stage1/stage1_step<N>.pt    ← saved every SAVE_EVERY steps
#     checkpoints/stage1/stage1_best.pt       ← best validation revenue
#     logs/training_log.csv                   ← step-by-step metrics
# """

# import os
# import sys
# import glob
# import math
# import csv
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from collections import deque
# from torch.optim import Adam
# from typing import Tuple

# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# from pipeline.config import *
# from pipeline.p3_models import TTFE, Actor, Critic, FeasibilityProjection

# # # ════════════════════════════════════════════════════════
# # # RUNNING REWARD NORMALISER
# # # ════════════════════════════════════════════════════════

# # class RunningNormaliser:
# #     """
# #     Tracks running mean and std of rewards using Welford's algorithm.
# #     Normalises rewards to roughly [-clip, +clip] range regardless of
# #     price spike magnitude. Far better than dividing by a fixed constant.
# #     """
# #     def __init__(self, clip: float = 10.0):
# #         self.mean  = 0.0
# #         self.var   = 1.0
# #         self.count = 0
# #         self.clip  = clip

# #     def update_and_normalise(self, reward: float) -> float:
# #         self.count += 1
# #         delta      = reward - self.mean
# #         self.mean += delta / self.count
# #         self.var  += (delta * (reward - self.mean) - self.var) / self.count
# #         std        = max(math.sqrt(abs(self.var)), 1e-6)
# #         return float(np.clip(reward / std, -self.clip, self.clip))

# # ════════════════════════════════════════════════════════
# # DATASET LOADER
# # ════════════════════════════════════════════════════════

# class ERCOTDataset:
#     """Loads merged parquets and serves rolling price windows + obs components."""

#     def __init__(self, split: str = "train"):
#         """
#         Args:
#             split: "train" (before VAL_START) or "val" (from VAL_START onward)
#         """
#         self.split = split
#         self.df    = self._load()
#         self.n     = len(self.df)

#         # Load normaliser stats saved by p2_build_dataset.py
#         stats_path = os.path.join(CHECKPOINT_DIR, "normaliser_stats.npz")
#         if not os.path.exists(stats_path):
#             raise FileNotFoundError(
#                 f"Normaliser not found at {stats_path}\n"
#                 "Run p2_build_dataset.py first."
#             )
#         stats       = np.load(stats_path, allow_pickle=True)
#         self.mean   = stats["mean"].astype(np.float32)   # shape: (19,) = 12+7
#         self.std    = stats["std"].astype(np.float32)

#         print(f"[Dataset:{split}] {self.n:,} rows | "
#               f"{self.df.index.min().date()} → {self.df.index.max().date()}")

#     # def _load(self) -> pd.DataFrame:
#     #     pattern = os.path.join(DATA_ROOT, "energy_prices", "*.parquet")
#     #     files   = sorted(glob.glob(pattern))
#     #     if not files:
#     #         raise FileNotFoundError("Run p0_download_data.py and p2_build_dataset.py first.")

#     #     # Load all three folders
#     #     def load_folder(subfolder):
#     #         fs = sorted(glob.glob(os.path.join(DATA_ROOT, subfolder, "*.parquet")))
#     #         parts = [pd.read_parquet(f) for f in fs]
#     #         df = pd.concat(parts, ignore_index=True)
#     #         if TIMESTAMP_COL != "__index__":
#     #             df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
#     #             df = df.set_index(TIMESTAMP_COL)
#     #         else:
#     #             df.index = pd.to_datetime(df.index)
#     #         return df.sort_index()
#     def _load(self) -> pd.DataFrame:
#         pattern = os.path.join(DATA_ROOT, "energy_prices", "*.parquet")
#         files   = sorted(glob.glob(pattern))
#         if not files:
#             raise FileNotFoundError("Run p0_download_data.py and p2_build_dataset.py first.")
#         def load_folder(subfolder):
#             fs = sorted(glob.glob(os.path.join(DATA_ROOT, subfolder, "*.parquet")))
#             parts = [pd.read_parquet(f) for f in fs]
#             df = pd.concat(parts)
#             if df.index.tz is not None:
#                 df.index = df.index.tz_localize(None)
#             return df.sort_index()

#         energy  = load_folder("energy_prices")
#         as_pr   = load_folder("as_prices")
#         syscond = load_folder("system_conditions")

#         df = energy.join(as_pr,   how="outer", rsuffix="_as")
#         df = df.join(syscond,     how="outer", rsuffix="_sys")
        
#         # Drop unused columns before dropna
#         cols_to_drop = [c for c in df.columns if
#                         c.startswith("rt_mcpc_") or
#                         c.startswith("is_post_rtcb")]
#         df = df.drop(columns=cols_to_drop, errors="ignore")

#         df = df.ffill(limit=3).dropna()
#         # Date split
#         if self.split == "train":
#             df = df[(df.index >= pd.Timestamp(STAGE1_START)) & (df.index < pd.Timestamp(VAL_START))]
#         else:
#             df = df[(df.index >= pd.Timestamp(VAL_START)) & (df.index <= pd.Timestamp(STAGE1_END))]

#         return df

#     def _normalise_price(self, raw: np.ndarray) -> np.ndarray:
#         """z-score + ±CLIP_SIGMA clip for price features."""
#         mean_p = self.mean[:PRICE_DIM]
#         std_p  = self.std[:PRICE_DIM]
#         return np.clip((raw - mean_p) / std_p, -CLIP_SIGMA, CLIP_SIGMA)

#     def _normalise_system(self, raw: np.ndarray) -> np.ndarray:
#         """z-score for system condition features."""
#         mean_s = self.mean[PRICE_DIM:]
#         std_s  = self.std[PRICE_DIM:]
#         return (raw - mean_s) / std_s

#     def get_price_window(self, idx: int) -> np.ndarray:
#         """Returns (WINDOW_LEN, PRICE_DIM) normalised price window ending at idx."""
#         start  = max(0, idx - WINDOW_LEN + 1)
#         window = self.df[PRICE_COLS].iloc[start:idx + 1].values.astype(np.float32)
#         if len(window) < WINDOW_LEN:
#             pad    = np.repeat(window[[0]], WINDOW_LEN - len(window), axis=0)
#             window = np.concatenate([pad, window], axis=0)
#         return self._normalise_price(window)

#     def get_system_vars(self, idx: int) -> np.ndarray:
#         """Returns (SYSTEM_DIM,) normalised system vars at idx."""
#         raw = self.df[SYSTEM_COLS].iloc[idx].values.astype(np.float32)
#         return self._normalise_system(raw)

#     @staticmethod
#     def time_features(ts: pd.Timestamp) -> np.ndarray:
#         """6-dim cyclical encoding: hour-of-day (×2) + day-of-week (×4)."""
#         h  = ts.hour + ts.minute / 60
#         dw = ts.dayofweek
#         return np.array([
#             math.sin(2 * math.pi * h  / 24),
#             math.cos(2 * math.pi * h  / 24),
#             math.sin(2 * math.pi * dw / 7),
#             math.cos(2 * math.pi * dw / 7),
#             math.sin(4 * math.pi * dw / 7),
#             math.cos(4 * math.pi * dw / 7),
#         ], dtype=np.float32)

#     def get_rt_lmp(self, idx: int) -> float:
#         return float(self.df[PRICE_COLS[0]].iloc[idx])

#     def get_timestamp(self, idx: int) -> pd.Timestamp:
#         return self.df.index[idx]

#     def __len__(self):
#         return self.n


# # ════════════════════════════════════════════════════════
# # ENVIRONMENT
# # ════════════════════════════════════════════════════════

# OBS_FLAT_DIM = WINDOW_LEN * PRICE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM


# class ERCOTEnv:
#     """
#     Steps through the dataset one 5-min interval at a time.
#     State  : (price_window, system_vars, time_feats, soc)
#     Action : charge/discharge rate ∈ [-1, 1]
#     Reward : energy arbitrage revenue for the interval ($)
#     """

#     def __init__(self, dataset: ERCOTDataset):
#         self.ds  = dataset
#         self.idx = WINDOW_LEN
#         self.soc = 0.5
#         self.ep_steps = 0 

#     # def reset(self):
#     #     self.idx = WINDOW_LEN
#     #     self.soc = 0.5
#     #     return self._obs()
#     # def reset(self):
#     #     max_start = int(len(self.ds) * 0.8)
#     #     self.idx  = np.random.randint(WINDOW_LEN, max_start)
#     #     self.soc  = np.random.uniform(0.3, 0.7)
#     #     return self._obs()
#     def reset(self):
#         max_start     = int(len(self.ds) * 0.8)
#         self.idx      = np.random.randint(WINDOW_LEN, max_start)
#         self.soc      = np.random.uniform(0.3, 0.7)
#         self.ep_steps = 0          # ← ADD THIS
#         return self._obs()
#     def _obs(self):
#         pw  = self.ds.get_price_window(self.idx)
#         sv  = self.ds.get_system_vars(self.idx)
#         tf  = ERCOTDataset.time_features(self.ds.get_timestamp(self.idx))
#         soc = np.array([self.soc], dtype=np.float32)
#         return pw, sv, tf, soc
    
#     def step(self, action: float, new_soc: float):
#         rt_lmp     = self.ds.get_rt_lmp(self.idx)
#         power_mw   = -action * BATTERY_POWER_MW
#         energy_mwh = power_mw * INTERVAL_H
#         reward     = (energy_mwh * rt_lmp) / REWARD_SCALE   # ← SCALED
#         self.soc       = new_soc
#         self.idx      += 1
#         self.ep_steps += 1                                   # ← ADD
#         done = (self.idx >= len(self.ds) - 1) or (self.ep_steps >= MAX_EP_STEPS)  # ← ADD
#         return self._obs(), reward, done

#     # def step(self, action: float, new_soc: float):
#     #     rt_lmp     = self.ds.get_rt_lmp(self.idx)
#     #     power_mw   = -action * BATTERY_POWER_MW    # +ve when discharging (selling)
#     #     energy_mwh = power_mw * INTERVAL_H
#     #     # reward     = energy_mwh * rt_lmp            # $/interval
#     #     # reward = (energy_mwh * rt_lmp) / 1000.0
#     #     reward = energy_mwh * rt_lmp

#     #     self.soc = new_soc
#     #     self.idx += 1
#     #     done = (self.idx >= len(self.ds) - 1)
#     #     return self._obs(), reward, done
    

# # ════════════════════════════════════════════════════════
# # REPLAY BUFFER
# # ════════════════════════════════════════════════════════

# class ReplayBuffer:
#     def __init__(self):
#         self.buf = deque(maxlen=REPLAY_SIZE)

#     def push(self, obs_flat, action, reward, nobs_flat, done):
#         self.buf.append((obs_flat, float(action), float(reward), nobs_flat, float(done)))

#     def sample(self) -> Tuple:
#         batch     = random.sample(self.buf, BATCH_SIZE)
#         obs, act, rew, nobs, done = zip(*batch)
#         return (
#             torch.FloatTensor(np.array(obs)).to(DEVICE),
#             torch.FloatTensor(act).unsqueeze(-1).to(DEVICE),
#             torch.FloatTensor(rew).unsqueeze(-1).to(DEVICE),
#             torch.FloatTensor(np.array(nobs)).to(DEVICE),
#             torch.FloatTensor(done).unsqueeze(-1).to(DEVICE),
#         )

#     def __len__(self):
#         return len(self.buf)


# # ════════════════════════════════════════════════════════
# # SAC AGENT
# # ════════════════════════════════════════════════════════

# def flatten_obs(pw, sv, tf, soc) -> np.ndarray:
#     return np.concatenate([pw.flatten(), sv, tf, soc])


# def unflatten_obs(flat: torch.Tensor) -> Tuple:
#     """Split flat buffer obs → (price_window, system_vars, time_feats, soc) tensors."""
#     B      = flat.shape[0]
#     pw_dim = WINDOW_LEN * PRICE_DIM
#     splits = torch.split(flat, [pw_dim, SYSTEM_DIM, TIME_DIM, SOC_DIM], dim=1)
#     pw     = splits[0].view(B, WINDOW_LEN, PRICE_DIM)
#     sv     = splits[1]
#     tf     = splits[2]
#     soc    = splits[3]
#     return pw, sv, tf, soc


# class SACAgent:
#     def __init__(self):
#         self.ttfe       = TTFE().to(DEVICE)
#         self.actor      = Actor().to(DEVICE)
#         self.critic     = Critic().to(DEVICE)
#         self.critic_tgt = Critic().to(DEVICE)
#         self.proj       = FeasibilityProjection().to(DEVICE)

#         # Hard-copy weights to target critic
#         self.critic_tgt.load_state_dict(self.critic.state_dict())
#         for p in self.critic_tgt.parameters():
#             p.requires_grad = False

#         # TTFE + actor share one optimiser (end-to-end)
#         self.opt_actor  = Adam(
#             list(self.ttfe.parameters()) + list(self.actor.parameters()),
#             lr=LR_ACTOR
#         )
#         self.opt_critic = Adam(self.critic.parameters(), lr=LR_CRITIC)

#         # Learnable entropy coefficient α
#         self.log_alpha  = torch.zeros(1, requires_grad=True, device=DEVICE)
#         self.opt_alpha  = Adam([self.log_alpha], lr=LR_ALPHA)
#         self.tgt_ent    = torch.tensor(TARGET_ENTROPY, device=DEVICE)

#     @property
#     def alpha(self):
#         return self.log_alpha.exp()

#     def encode(self, pw: torch.Tensor, sv: torch.Tensor,
#                tf: torch.Tensor, soc: torch.Tensor) -> torch.Tensor:
#         feat = self.ttfe(pw)
#         return torch.cat([feat, sv, tf, soc], dim=-1)

#     def select_action(self, pw, sv, tf, soc_val: float,
#                       deterministic: bool = False) -> Tuple[float, float]:
#         pw_t  = torch.FloatTensor(pw).unsqueeze(0).to(DEVICE)
#         sv_t  = torch.FloatTensor(sv).unsqueeze(0).to(DEVICE)
#         tf_t  = torch.FloatTensor(tf).unsqueeze(0).to(DEVICE)
#         soc_t = torch.FloatTensor([[soc_val]]).to(DEVICE)

#         with torch.no_grad():
#             obs = self.encode(pw_t, sv_t, tf_t, soc_t)
#             if deterministic:
#                 raw = self.actor.get_deterministic_action(obs)
#             else:
#                 raw, _ = self.actor.sample(obs)
#             feasible, new_soc = self.proj(raw, soc_t)
#         return feasible.item(), new_soc.item()

#     def update(self, buffer: ReplayBuffer) -> dict:
#         if len(buffer) < BATCH_SIZE:
#             return {}

#         obs_flat, act, rew, nobs_flat, done = buffer.sample()

#         # Unflatten and encode observations
#         pw,  sv,  tf,  soc  = unflatten_obs(obs_flat)
#         npw, nsv, ntf, nsoc = unflatten_obs(nobs_flat)
#         obs_enc  = self.encode(pw,  sv,  tf,  soc)
#         nobs_enc = self.encode(npw, nsv, ntf, nsoc)

#         # ── Critic update ────────────────────────────────────────────
#         with torch.no_grad():
#             next_act, next_lp = self.actor.sample(nobs_enc)
#             q_tgt = self.critic_tgt.q_min(nobs_enc, next_act)
#             y     = rew + GAMMA * (1 - done) * (q_tgt - self.alpha * next_lp)

#         q1, q2      = self.critic(obs_enc, act)
#         critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

#         # self.opt_critic.zero_grad()
#         # critic_loss.backward()
#         # self.opt_critic.step()
#         self.opt_critic.zero_grad()
#         critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
#         self.opt_critic.step()

#         # ── Actor + TTFE update ──────────────────────────────────────
#         # Re-encode (detach critic from this pass)
#         obs_enc2 = self.encode(pw, sv, tf, soc)
#         new_act, log_pi = self.actor.sample(obs_enc2)
#         q_new = self.critic.q_min(obs_enc2, new_act)
#         actor_loss = (self.alpha.detach() * log_pi - q_new).mean()

#         # self.opt_actor.zero_grad()
#         # actor_loss.backward()
#         # self.opt_actor.step()
#         self.opt_actor.zero_grad()
#         actor_loss.backward()
#         torch.nn.utils.clip_grad_norm_(list(self.ttfe.parameters()) + list(self.actor.parameters()),
#                                        max_norm=1.0
#                                        )
#         self.opt_actor.step()

#         # ── Alpha update ─────────────────────────────────────────────
#         alpha_loss = -(self.log_alpha * (log_pi + self.tgt_ent).detach()).mean()
#         self.opt_alpha.zero_grad()
#         alpha_loss.backward()
#         self.opt_alpha.step()

#         # ── Soft target update ────────────────────────────────────────
#         for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
#             pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

#         return {
#             "critic_loss": critic_loss.item(),
#             "actor_loss":  actor_loss.item(),
#             "alpha":       self.alpha.item(),
#         }

#     def save(self, step: int, tag: str = ""):
#         os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#         fname = f"stage1_{tag or f'step{step}'}.pt"
#         torch.save({
#             "step":       step,
#             "ttfe":       self.ttfe.state_dict(),
#             "actor":      self.actor.state_dict(),
#             "critic":     self.critic.state_dict(),
#             "critic_tgt": self.critic_tgt.state_dict(),
#             "log_alpha":  self.log_alpha.data,
#         }, os.path.join(CHECKPOINT_DIR, fname))
#         print(f"  [Saved] {fname}")

#     def load(self, path: str):
#         ckpt = torch.load(path, map_location=DEVICE)
#         self.ttfe.load_state_dict(ckpt["ttfe"])
#         self.actor.load_state_dict(ckpt["actor"])
#         self.critic.load_state_dict(ckpt["critic"])
#         self.critic_tgt.load_state_dict(ckpt["critic_tgt"])
#         self.log_alpha.data = ckpt["log_alpha"]
#         print(f"  [Loaded] step={ckpt['step']} from {path}")
#         return ckpt["step"]


# # ════════════════════════════════════════════════════════
# # QUICK VALIDATION (deterministic rollout on val split)
# # ════════════════════════════════════════════════════════

# def quick_val(agent: SACAgent, val_dataset: ERCOTDataset,
#               max_steps: int = 2000) -> float:
#     """Returns total revenue over first max_steps of val split."""
#     env   = ERCOTEnv(val_dataset)
#     obs   = env.reset()
#     pw, sv, tf, soc_arr = obs
#     soc_val = float(soc_arr[0])
#     total_rev = 0.0

#     for _ in range(max_steps):
#         action, new_soc = agent.select_action(pw, sv, tf, soc_val, deterministic=True)
#         (pw, sv, tf, soc_arr), reward, done = env.step(action, new_soc)
#         soc_val   = float(soc_arr[0])
#         total_rev += reward
#         if done:
#             break

#     return total_rev

# # ADD this function before main():
# def collect_demonstrations(env: ERCOTEnv, dataset: ERCOTDataset,
#                            buffer: ReplayBuffer, n_steps: int = 10000):
#     """
#     Fill replay buffer with rule-based heuristic transitions.
#     Charge when rt_lmp < median, discharge when rt_lmp >= median.
#     This bootstraps the critic with meaningful value estimates.
#     """
#     print(f"[Demo] Collecting {n_steps} rule-based demonstrations...")
#     median_price = float(dataset.df[PRICE_COLS[0]].median())
#     obs = env.reset()
#     pw, sv, tf, soc_arr = obs
#     soc_val = float(soc_arr[0])

#     for i in range(n_steps):
#         rt_lmp = dataset.get_rt_lmp(env.idx)
#         # Rule: charge below median, discharge above
#         raw_action = -1.0 if rt_lmp >= median_price else 1.0
#         # Apply feasibility
#         a_t = torch.FloatTensor([[raw_action]]).to(DEVICE)
#         s_t = torch.FloatTensor([[soc_val]]).to(DEVICE)
#         with torch.no_grad():
#             proj = FeasibilityProjection().to(DEVICE)
#             _, ns_t = proj(a_t, s_t)
#         action  = raw_action
#         new_soc = ns_t.item()

#         next_obs, reward, done = env.step(action, new_soc)
#         npw, nsv, ntf, nsoc_arr = next_obs

#         obs_flat  = flatten_obs(pw,  sv,  tf,  soc_arr)
#         nobs_flat = flatten_obs(npw, nsv, ntf, nsoc_arr)
#         buffer.push(obs_flat, action, reward, nobs_flat, float(done))
#         # scaled_reward = reward / REWARD_SCALE
#         # buffer.push(obs_flat, action, scaled_reward, nobs_flat, float(done))

#         pw, sv, tf, soc_arr = next_obs
#         soc_val = float(nsoc_arr[0])

#         if done:
#             obs = env.reset()
#             pw, sv, tf, soc_arr = obs
#             soc_val = float(soc_arr[0])

#     print(f"[Demo] Buffer filled with {len(buffer)} transitions")

# # ════════════════════════════════════════════════════════
# # MAIN TRAINING LOOP
# # ════════════════════════════════════════════════════════

# def main():
#     print("=" * 60)
#     print("Pipeline 4 — Stage 1 SAC Training")
#     print(f"  Device      : {DEVICE}")
#     print(f"  Total steps : {TOTAL_STEPS:,}")
#     print("=" * 60)

#     os.makedirs(LOG_DIR, exist_ok=True)
#     os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#     # Data
#     train_ds  = ERCOTDataset("train")
#     val_ds    = ERCOTDataset("val")
#     train_env = ERCOTEnv(train_ds)

#     # Agent + buffer
#     agent  = SACAgent()
#     buffer = ReplayBuffer()

#     # ── Bootstrap buffer with rule-based demonstrations ──────────
#     collect_demonstrations(train_env, train_ds, buffer, n_steps=10000)

#     # Log file
#     log_path = os.path.join(LOG_DIR, "training_log.csv")
#     log_file = open(log_path, "w", newline="")
#     writer   = csv.writer(log_file)
#     writer.writerow(["step", "critic_loss", "actor_loss", "alpha",
#                      "episode_reward", "val_revenue"])

#     # State
#     obs_parts = train_env.reset()
#     pw, sv, tf, soc_arr = obs_parts
#     soc_val   = float(soc_arr[0])
#     ep_reward = 0.0
#     best_val  = -float("inf")
#     recent_losses = {"critic_loss": [], "actor_loss": [], "alpha": []}

#     print(f"\nStarting training loop...\n")

#     for step in range(1, TOTAL_STEPS + 1):

#         # ── Select action (no warmup — buffer pre-filled) ──────────
#         action, new_soc = agent.select_action(pw, sv, tf, soc_val)

#         # ── Environment step ───────────────────────────────────────
#         next_obs, reward, done = train_env.step(action, new_soc)
#         npw, nsv, ntf, nsoc_arr = next_obs

#         # ── Store transition (reward already scaled in env.step) ───
#         obs_flat  = flatten_obs(pw,  sv,  tf,  soc_arr)
#         nobs_flat = flatten_obs(npw, nsv, ntf, nsoc_arr)
#         buffer.push(obs_flat, action, reward, nobs_flat, float(done))

#         ep_reward += reward * REWARD_SCALE   # track real dollars

#         # ── SAC update ─────────────────────────────────────────────
#         info = agent.update(buffer)
#         for k, v in info.items():
#             recent_losses[k].append(v)

#         # ── Advance state ──────────────────────────────────────────
#         pw, sv, tf, soc_arr = next_obs
#         soc_val = float(nsoc_arr[0])

#         # ── Episode reset ──────────────────────────────────────────
#         if done:
#             obs_parts = train_env.reset()
#             pw, sv, tf, soc_arr = obs_parts
#             soc_val   = float(soc_arr[0])
#             ep_reward = 0.0

#         # ── Logging ────────────────────────────────────────────────
#         if step % LOG_EVERY == 0:
#             cl  = np.mean(recent_losses["critic_loss"][-100:]) if recent_losses["critic_loss"] else 0
#             al  = np.mean(recent_losses["actor_loss"][-100:])  if recent_losses["actor_loss"]  else 0
#             alp = np.mean(recent_losses["alpha"][-100:])       if recent_losses["alpha"]        else 0
#             print(f"  step={step:>7,} | critic={cl:.4f} | actor={al:.4f} | alpha={alp:.4f}")

#         # ── Validation ─────────────────────────────────────────────
#         if step % EVAL_EVERY == 0:
#             val_rev = quick_val(agent, val_ds) * REWARD_SCALE  # display real dollars
#             print(f"  ★ Val revenue (2000 steps): ${val_rev:+,.2f}")
#             writer.writerow([step, cl, al, alp, ep_reward, val_rev])
#             log_file.flush()

#             if val_rev > best_val:
#                 best_val = val_rev
#                 agent.save(step, tag="best")
#                 print(f"  ↑ New best: ${best_val:+,.2f}")

#         # ── Checkpoint ─────────────────────────────────────────────
#         if step % SAVE_EVERY == 0:
#             agent.save(step)

#     # Final save
#     agent.save(TOTAL_STEPS, tag="final")
#     log_file.close()

#     print("\n" + "=" * 60)
#     print("✓ Training complete.")
#     print(f"  Best val revenue : ${best_val:+,.2f}")
#     print(f"  Logs saved       : {log_path}")
#     print("Next step:  python pipeline/p5_evaluate.py")


# if __name__ == "__main__":
#     main()
"""
Pipeline 4 — SAC Training Loop (Improved Run)
===============================================
Changes from Plan A (previous run):
  1. Dataset: 2022-2025 only (excludes 2021 Winter Storm Uri outlier)
  2. Huber loss for critic (replaces MSE — reduces spike gradient dominance)
  3. Two replay buffers: demo_buffer + agent_buffer with decaying demo ratio
  4. Demo ratio never decays to 0 (floor at DEMO_FLOOR=5%)
  5. LR_CRITIC = 1e-4 (from 3e-4) — slower, more stable critic updates
  6. GRAD_CLIP = 0.5 (from 1.0) — tighter clipping
  7. Health-gated checkpointing — only save when critic stable AND log_pi < 0
  8. Early stopping — critic > 300 OR log_pi > 0 triggers stop with save
  9. Deterministic quick_val — reproducible evaluation every EVAL_EVERY steps
  10. Richer logging — log_pi + charge_fraction tracked every LOG_EVERY steps

Run AFTER p2_build_dataset.py (MUST rerun if STAGE1_START changed to 2022-01-01).

Usage:
    python pipeline/p4_train.py

Outputs:
    checkpoints/stage1/stage1_step<N>.pt      ← every SAVE_EVERY steps
    checkpoints/stage1/stage1_best.pt         ← best HEALTHY val revenue
    checkpoints/stage1/stage1_emergency.pt    ← saved at early stop
    logs/training_log.csv                     ← step-by-step metrics
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
                "IMPORTANT: Rerun p2 if STAGE1_START changed to 2022-01-01."
            )
        stats     = np.load(stats_path, allow_pickle=True)
        self.mean = stats["mean"].astype(np.float32)
        self.std  = stats["std"].astype(np.float32)

        print(f"[Dataset:{split}] {self.n:,} rows | "
              f"{self.df.index.min().date()} → {self.df.index.max().date()}")

    def _load(self) -> pd.DataFrame:
        pattern = os.path.join(DATA_ROOT, "energy_prices", "*.parquet")
        files   = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError("Run p0_download_data.py and p2_build_dataset.py first.")

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
        else:
            df = df[(df.index >= pd.Timestamp(VAL_START)) &
                    (df.index <= pd.Timestamp(STAGE1_END))]
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
# ENVIRONMENT
# ════════════════════════════════════════════════════════

OBS_FLAT_DIM = WINDOW_LEN * PRICE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM


class ERCOTEnv:
    """
    Steps through the dataset one 5-min interval at a time.
    State  : (price_window, system_vars, time_feats, soc)
    Action : charge/discharge rate ∈ [-1, 1]  (+1=charge, -1=discharge)
    Reward : energy arbitrage revenue / REWARD_SCALE
    """

    def __init__(self, dataset: ERCOTDataset):
        self.ds       = dataset
        self.idx      = WINDOW_LEN
        self.soc      = 0.5
        self.ep_steps = 0

    def reset(self):
        max_start     = int(len(self.ds) * 0.8)
        self.idx      = np.random.randint(WINDOW_LEN, max_start)
        self.soc      = np.random.uniform(0.3, 0.7)
        self.ep_steps = 0
        return self._obs()

    def _obs(self):
        pw  = self.ds.get_price_window(self.idx)
        sv  = self.ds.get_system_vars(self.idx)
        tf  = ERCOTDataset.time_features(self.ds.get_timestamp(self.idx))
        soc = np.array([self.soc], dtype=np.float32)
        return pw, sv, tf, soc

    def step(self, action: float, new_soc: float):
        rt_lmp     = self.ds.get_rt_lmp(self.idx)
        power_mw   = -action * BATTERY_POWER_MW
        energy_mwh = power_mw * INTERVAL_H
        reward     = (energy_mwh * rt_lmp) / REWARD_SCALE

        self.soc       = new_soc
        self.idx      += 1
        self.ep_steps += 1
        done = (self.idx >= len(self.ds) - 1) or (self.ep_steps >= MAX_EP_STEPS)
        return self._obs(), reward, done


# ════════════════════════════════════════════════════════
# REPLAY BUFFERS (two separate buffers)
# ════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Replay buffer with variable capacity and variable sample size.
    Used for both demo_buffer and agent_buffer.
    """
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
    Linearly decay demo sampling ratio from 1.0 to DEMO_FLOOR over DEMO_DECAY_STEPS.

    Key design decision: floor at DEMO_FLOOR (5%), never 0.
    Rationale: previous run showed policy collapsed within 10-20k steps.
    Once collapsed, agent_buffer fills with always-discharge transitions.
    Keeping 5% demo ensures critic always sees charge/discharge diversity.

    Example:
        step=0:          ratio = 1.00  (pure demos)
        step=100,000:    ratio = 0.53
        step=200,000:    ratio = 0.05  (floor, stays here)
        step=300,000+:   ratio = 0.05  (floor maintained)
    """
    ratio = 1.0 - (step / DEMO_DECAY_STEPS) * (1.0 - DEMO_FLOOR)
    return max(DEMO_FLOOR, ratio)


def sample_mixed(demo_buf: ReplayBuffer,
                 agent_buf: ReplayBuffer,
                 step: int) -> Optional[Tuple]:
    """
    Sample a mixed batch from demo_buffer and agent_buffer.

    Returns None if insufficient data (training should skip this step).
    Automatically adjusts n_demo and n_agent if buffers are smaller than requested.
    """
    demo_ratio = get_demo_ratio(step)
    n_demo     = int(BATCH_SIZE * demo_ratio)
    n_agent    = BATCH_SIZE - n_demo

    # Clamp to what's available
    n_demo  = min(n_demo,  len(demo_buf))
    n_agent = min(n_agent, len(agent_buf))

    if n_demo + n_agent < 64:   # minimum viable batch
        return None

    parts = []
    if n_demo > 0:
        parts.append(demo_buf.sample(n_demo))
    if n_agent > 0:
        parts.append(agent_buf.sample(n_agent))

    if len(parts) == 1:
        return parts[0]

    # Concatenate along batch dimension
    return tuple(torch.cat([p[i] for p in parts], dim=0) for i in range(5))


# ════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════

def flatten_obs(pw, sv, tf, soc) -> np.ndarray:
    return np.concatenate([pw.flatten(), sv, tf, soc])


def unflatten_obs(flat: torch.Tensor) -> Tuple:
    """Split flat buffer obs → (price_window, system_vars, time_feats, soc)."""
    B      = flat.shape[0]
    pw_dim = WINDOW_LEN * PRICE_DIM
    splits = torch.split(flat, [pw_dim, SYSTEM_DIM, TIME_DIM, SOC_DIM], dim=1)
    pw     = splits[0].view(B, WINDOW_LEN, PRICE_DIM)
    sv     = splits[1]
    tf     = splits[2]
    soc    = splits[3]
    return pw, sv, tf, soc


# ════════════════════════════════════════════════════════
# SAC AGENT
# ════════════════════════════════════════════════════════

class SACAgent:
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

        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.opt_alpha = Adam([self.log_alpha], lr=LR_ALPHA)
        self.tgt_ent   = torch.tensor(TARGET_ENTROPY, device=DEVICE)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def encode(self, pw, sv, tf, soc) -> torch.Tensor:
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

    def update(self, batch: Tuple) -> dict:
        """
        Single SAC update step using a pre-sampled mixed batch.

        Key changes from Plan A:
          - Huber loss replaces MSE for critic (reduces spike gradient dominance)
          - LR_CRITIC = 1e-4 (more conservative critic updates)
          - GRAD_CLIP = 0.5 (tighter clipping)
          - Returns log_pi for health monitoring
        """
        obs_flat, act, rew, nobs_flat, done = batch

        pw,  sv,  tf,  soc  = unflatten_obs(obs_flat)
        npw, nsv, ntf, nsoc = unflatten_obs(nobs_flat)
        obs_enc  = self.encode(pw,  sv,  tf,  soc)
        nobs_enc = self.encode(npw, nsv, ntf, nsoc)

        # ── Critic update ─────────────────────────────────────────────
        with torch.no_grad():
            next_act, next_lp = self.actor.sample(nobs_enc)
            q_tgt = self.critic_tgt.q_min(nobs_enc, next_act)
            y     = rew + GAMMA * (1 - done) * (q_tgt - self.alpha * next_lp)

        q1, q2 = self.critic(obs_enc, act)

        # Huber loss instead of MSE:
        #   For |error| < HUBER_DELTA (=10): loss = error² / 2  (quadratic, like MSE)
        #   For |error| >= HUBER_DELTA:       loss = delta * |error| (linear — caps spikes)
        # This reduces spike gradient dominance from 40,580x to ~15x
        critic_loss = (F.huber_loss(q1, y, delta=HUBER_DELTA) +
                       F.huber_loss(q2, y, delta=HUBER_DELTA))

        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP)
        self.opt_critic.step()

        # ── Actor + TTFE update ───────────────────────────────────────
        obs_enc2 = self.encode(pw, sv, tf, soc)
        new_act, log_pi = self.actor.sample(obs_enc2)
        q_new           = self.critic.q_min(obs_enc2, new_act)
        actor_loss      = (self.alpha.detach() * log_pi - q_new).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.ttfe.parameters()) + list(self.actor.parameters()),
            max_norm=GRAD_CLIP
        )
        self.opt_actor.step()

        # ── Alpha update ──────────────────────────────────────────────
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
            "log_pi":      log_pi.mean().item(),  # key health indicator
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
# QUICK VALIDATION — deterministic rollout
# ════════════════════════════════════════════════════════

def quick_val(agent: SACAgent, val_dataset: ERCOTDataset,
              max_steps: int = 2000) -> float:
    """
    Deterministic rollout on val split.
    Always starts from idx=WINDOW_LEN, SoC=0.5 for reproducibility.

    Previous issue: random starts caused 10x revenue variance across runs.
    This version gives stable, comparable numbers every evaluation.
    """
    env           = ERCOTEnv(val_dataset)
    env.idx       = WINDOW_LEN   # deterministic start
    env.soc       = 0.5
    env.ep_steps  = 0
    pw, sv, tf, soc_arr = env._obs()
    soc_val       = float(soc_arr[0])
    total_rev     = 0.0

    for _ in range(max_steps):
        action, new_soc = agent.select_action(pw, sv, tf, soc_val, deterministic=True)
        (pw, sv, tf, soc_arr), reward, done = env.step(action, new_soc)
        soc_val    = float(soc_arr[0])
        total_rev += reward
        if done:
            break

    return total_rev


# ════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ════════════════════════════════════════════════════════

def collect_demonstrations(env: ERCOTEnv, dataset: ERCOTDataset,
                            buffer: ReplayBuffer, n_steps: int = DEMO_STEPS):
    """
    Fill demo_buffer with rule-based heuristic transitions.
    Charge when rt_lmp < median, discharge when rt_lmp >= median.

    Using training dataset median (not val median) — no look-ahead.
    DEMO_STEPS = 50,000 → ~173 diverse episodes (vs 34 previously with 10k)
    This gives the critic a much richer prior over market conditions.
    """
    print(f"[Demo] Collecting {n_steps:,} rule-based demonstrations...")
    median_price = float(dataset.df[PRICE_COLS[0]].median())
    print(f"[Demo] Training median price: ${median_price:.2f}")

    obs = env.reset()
    pw, sv, tf, soc_arr = obs
    soc_val = float(soc_arr[0])
    proj    = FeasibilityProjection().to(DEVICE)

    charge_count   = 0
    discharge_count = 0

    for i in range(n_steps):
        rt_lmp     = dataset.get_rt_lmp(env.idx)
        raw_action = -1.0 if rt_lmp >= median_price else 1.0

        a_t = torch.FloatTensor([[raw_action]]).to(DEVICE)
        s_t = torch.FloatTensor([[soc_val]]).to(DEVICE)
        with torch.no_grad():
            _, ns_t = proj(a_t, s_t)
        new_soc = ns_t.item()

        next_obs, reward, done = env.step(raw_action, new_soc)
        npw, nsv, ntf, nsoc_arr = next_obs

        obs_flat  = flatten_obs(pw,  sv,  tf,  soc_arr)
        nobs_flat = flatten_obs(npw, nsv, ntf, nsoc_arr)
        buffer.push(obs_flat, raw_action, reward, nobs_flat, float(done))

        if raw_action > 0:
            charge_count += 1
        else:
            discharge_count += 1

        pw, sv, tf, soc_arr = next_obs
        soc_val = float(nsoc_arr[0])

        if done:
            obs = env.reset()
            pw, sv, tf, soc_arr = obs
            soc_val = float(soc_arr[0])

        if (i + 1) % 10_000 == 0:
            print(f"[Demo] {i+1:,}/{n_steps:,} steps collected...")

    total = charge_count + discharge_count
    print(f"[Demo] Complete: {len(buffer):,} transitions in buffer")
    print(f"[Demo] Action balance: {charge_count/total*100:.1f}% charge, "
          f"{discharge_count/total*100:.1f}% discharge")
    print(f"[Demo] (Expected: ~50% each — confirms demos cover both actions)")


# ════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Pipeline 4 — Stage 1 SAC Training (Improved Run)")
    print(f"  Device        : {DEVICE}")
    print(f"  Dataset       : {STAGE1_START} → {VAL_START} (excludes 2021)")
    print(f"  Total steps   : {TOTAL_STEPS:,}")
    print(f"  LR_CRITIC     : {LR_CRITIC}  (from 3e-4)")
    print(f"  GRAD_CLIP     : {GRAD_CLIP}  (from 1.0)")
    print(f"  HUBER_DELTA   : {HUBER_DELTA}")
    print(f"  DEMO_STEPS    : {DEMO_STEPS:,}  (from 10k)")
    print(f"  DEMO_FLOOR    : {DEMO_FLOOR}  (never goes to 0)")
    print(f"  DEMO_DECAY    : {DEMO_DECAY_STEPS:,} steps")
    print(f"  Early stop    : critic>{CRITIC_LOSS_STOP} OR log_pi>{LOG_PI_STOP}"
          f" after step {MIN_STEP_BEFORE_STOP:,}")
    print("=" * 65)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────
    train_ds  = ERCOTDataset("train")
    val_ds    = ERCOTDataset("val")
    train_env = ERCOTEnv(train_ds)

    print(f"\n  Training rows : {len(train_ds):,}")
    print(f"  Val rows      : {len(val_ds):,}")

    # ── Agent and two replay buffers ───────────────────────────────
    agent       = SACAgent()
    demo_buffer = ReplayBuffer(capacity=DEMO_BUFFER_SIZE)
    agent_buffer= ReplayBuffer(capacity=AGENT_BUFFER_SIZE)

    # ── Fill demo buffer with DEMO_STEPS rule-based transitions ───
    collect_demonstrations(train_env, train_ds, demo_buffer, n_steps=DEMO_STEPS)

    # ── Log file ───────────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["step", "critic_loss", "actor_loss", "alpha", "log_pi",
                     "charge_frac", "demo_ratio", "val_revenue"])

    # ── Training state ─────────────────────────────────────────────
    obs_parts = train_env.reset()
    pw, sv, tf, soc_arr = obs_parts
    soc_val    = float(soc_arr[0])
    ep_reward  = 0.0
    best_val   = -float("inf")

    recent_losses  = {
        "critic_loss": deque(maxlen=100),
        "actor_loss":  deque(maxlen=100),
        "alpha":       deque(maxlen=100),
        "log_pi":      deque(maxlen=100),
    }
    recent_actions = deque(maxlen=1000)   # for charge fraction tracking

    # ── Variables for logging (initialise before first LOG_EVERY) ─
    cl = al = alp = alp_lp = charge_frac = 0.0

    print(f"\nStarting training loop...\n")
    stop_training = False

    for step in range(1, TOTAL_STEPS + 1):

        # ── Select action ──────────────────────────────────────────
        action, new_soc = agent.select_action(pw, sv, tf, soc_val)
        recent_actions.append(action)

        # ── Environment step ───────────────────────────────────────
        next_obs, reward, done = train_env.step(action, new_soc)
        npw, nsv, ntf, nsoc_arr = next_obs

        # ── Store in agent buffer ──────────────────────────────────
        obs_flat  = flatten_obs(pw,  sv,  tf,  soc_arr)
        nobs_flat = flatten_obs(npw, nsv, ntf, nsoc_arr)
        agent_buffer.push(obs_flat, action, reward, nobs_flat, float(done))

        ep_reward += reward * REWARD_SCALE

        # ── Mixed batch update ─────────────────────────────────────
        batch = sample_mixed(demo_buffer, agent_buffer, step)
        if batch is not None:
            info = agent.update(batch)
            for k in recent_losses:
                if k in info:
                    recent_losses[k].append(info[k])

        # ── Advance state ──────────────────────────────────────────
        pw, sv, tf, soc_arr = next_obs
        soc_val = float(nsoc_arr[0])

        if done:
            obs_parts = train_env.reset()
            pw, sv, tf, soc_arr = obs_parts
            soc_val   = float(soc_arr[0])
            ep_reward = 0.0

        # ── Logging ────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            cl   = float(np.mean(recent_losses["critic_loss"])) if recent_losses["critic_loss"] else 0.0
            al   = float(np.mean(recent_losses["actor_loss"]))  if recent_losses["actor_loss"]  else 0.0
            alp  = float(np.mean(recent_losses["alpha"]))       if recent_losses["alpha"]        else 0.0
            alp_lp = float(np.mean(recent_losses["log_pi"]))   if recent_losses["log_pi"]       else 0.0
            charge_frac = sum(1 for a in recent_actions if a > 0) / max(len(recent_actions), 1)
            demo_ratio  = get_demo_ratio(step)

            print(f"  step={step:>7,} | critic={cl:.2f} | actor={al:.2f} | "
                  f"alpha={alp:.4f} | log_pi={alp_lp:+.3f} | "
                  f"charge%={charge_frac*100:.1f} | demo%={demo_ratio*100:.0f}")

            # Charge fraction warning
            if charge_frac < CHARGE_FRAC_MIN and step > 10_000:
                print(f"  [WARN] Charge fraction {charge_frac*100:.1f}% < {CHARGE_FRAC_MIN*100:.0f}% "
                      f"— possible policy collapse beginning")

        # ── Validation ─────────────────────────────────────────────
        if step % EVAL_EVERY == 0:
            val_rev = quick_val(agent, val_ds) * REWARD_SCALE
            demo_ratio = get_demo_ratio(step)
            print(f"  ★ Val revenue (2000 steps): ${val_rev:+,.2f}  "
                  f"[critic={cl:.1f} | log_pi={alp_lp:+.3f}]")

            writer.writerow([step, cl, al, alp, alp_lp, charge_frac,
                             demo_ratio, val_rev])
            log_file.flush()

            # Health-gated checkpoint saving
            # Only save best if critic is stable AND policy not collapsed
            is_healthy = (cl < CRITIC_LOSS_STOP) and (alp_lp < LOG_PI_STOP)

            if val_rev > best_val and is_healthy:
                best_val = val_rev
                agent.save(step, tag="best")
                print(f"  ↑ New best (healthy checkpoint): ${best_val:+,.2f}")
            elif val_rev > best_val and not is_healthy:
                print(f"  [SKIP] Val ${val_rev:+,.2f} > best but checkpoint unhealthy "
                      f"(critic={cl:.1f} > {CRITIC_LOSS_STOP} OR "
                      f"log_pi={alp_lp:+.3f} > {LOG_PI_STOP})")

        # ── Early stopping ─────────────────────────────────────────
        if step > MIN_STEP_BEFORE_STOP and recent_losses["critic_loss"]:
            cl_check = float(np.mean(recent_losses["critic_loss"]))
            lp_check = float(np.mean(recent_losses["log_pi"])) if recent_losses["log_pi"] else -1.0

            if cl_check > CRITIC_LOSS_STOP:
                print(f"\n  [EARLY STOP] Critic loss {cl_check:.1f} > {CRITIC_LOSS_STOP}"
                      f" at step {step}. Saving emergency checkpoint.")
                agent.save(step, tag="emergency")
                stop_training = True

            elif lp_check > LOG_PI_STOP and step > 20_000:
                print(f"\n  [EARLY STOP] log_pi {lp_check:+.3f} > {LOG_PI_STOP}"
                      f" at step {step}. Policy has collapsed. Saving emergency checkpoint.")
                agent.save(step, tag="emergency")
                stop_training = True

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
    print(f"  Stopped at step      : {step:,}")
    print(f"  Best val revenue     : ${best_val:+,.2f}")
    print(f"  Early stopped        : {stop_training}")
    print(f"  Logs saved           : {log_path}")
    print("Next step:  python pipeline/p5_evaluate.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
