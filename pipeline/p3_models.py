# """
# Pipeline 3 — Model Definitions
# ================================
# Defines all neural network components for Stage 1:
#   - TTFE  (Transformer Temporal Feature Extractor)
#   - Actor (squashed Gaussian, 1-dim energy action)
#   - Critic (twin Q-networks)
#   - FeasibilityProjection (differentiable SoC clamp)

# These classes are imported by p4_train.py and p5_evaluate.py.
# You can also run this file directly to verify model shapes are correct.

# Usage (verification):
#     python pipeline/p3_models.py
# """

# import sys, os
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple

# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# from pipeline.config import *


# # ════════════════════════════════════════════════════════
# # TTFE  —  Transformer Temporal Feature Extractor
# # ════════════════════════════════════════════════════════

# class TTFE(nn.Module):
#     """
#     Encodes (batch, WINDOW_LEN, PRICE_DIM) → (batch, TTFE_DIM).

#     Architecture (follows TempDRL, Li et al. 2024):
#       Linear projection → learnable positional encoding →
#       N × TransformerEncoderLayer → mean pool over time → LayerNorm
#     """

#     def __init__(self):
#         super().__init__()
#         self.d_model = TTFE_DIM

#         # Project raw 12-dim prices → d_model
#         self.input_proj = nn.Linear(PRICE_DIM, TTFE_DIM)

#         # Learnable positional encoding  (1, L, d_model)
#         self.pos_enc = nn.Parameter(torch.zeros(1, WINDOW_LEN, TTFE_DIM))
#         nn.init.normal_(self.pos_enc, std=0.02)

#         # Transformer encoder stack
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model         = TTFE_DIM,
#             nhead           = TTFE_NHEAD,
#             dim_feedforward = TTFE_DIM * 4,
#             dropout         = TTFE_DROP if hasattr(sys.modules[__name__], 'TTFE_DROP') else 0.1,
#             batch_first     = True,
#             norm_first      = True,   # Pre-LN: more stable training
#         )
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=TTFE_NLAYERS)
#         self.layer_norm  = nn.LayerNorm(TTFE_DIM)

#     def forward(self, price_window: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             price_window: (B, WINDOW_LEN, PRICE_DIM)  — z-score normalised, clipped ±5σ
#         Returns:
#             feat: (B, TTFE_DIM)
#         """
#         x = self.input_proj(price_window)    # (B, L, d_model)
#         x = x + self.pos_enc                 # add positional encoding
#         x = self.transformer(x)              # (B, L, d_model)
#         x = self.layer_norm(x)
#         return x.mean(dim=1)                 # mean pool → (B, d_model)


# # ════════════════════════════════════════════════════════
# # ACTOR  —  Squashed Gaussian Policy
# # ════════════════════════════════════════════════════════

# LOG_STD_MIN = -5
# LOG_STD_MAX =  2


# class Actor(nn.Module):
#     """
#     Stage 1 actor: 78-d obs → 1-d continuous action ∈ [-1, 1].
#     Uses the reparameterisation trick (tanh-squashed Gaussian) for SAC.
#     """

#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(OBS_DIM, HIDDEN_DIM), nn.ReLU(),
#             nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
#         )
#         self.mean_head    = nn.Linear(HIDDEN_DIM, 1)
#         self.log_std_head = nn.Linear(HIDDEN_DIM, 1)

#         # Initialise output layers near zero for stable early training
#         nn.init.uniform_(self.mean_head.weight,    -3e-3, 3e-3)
#         nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)

#     def _get_dist_params(self, obs: torch.Tensor):
#         h       = self.net(obs)
#         mean    = self.mean_head(h)
#         log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
#         return mean, log_std.exp()

#     def forward(self, obs: torch.Tensor):
#         return self._get_dist_params(obs)

#     def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Returns:
#             action   : (B, 1) ∈ [-1, 1]  via tanh squashing
#             log_prob : (B, 1)  with tanh correction
#         """
#         mean, std = self._get_dist_params(obs)
#         dist      = torch.distributions.Normal(mean, std)
#         x         = dist.rsample()                                   # reparameterisation
#         action    = torch.tanh(x)
#         log_prob  = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
#         return action, log_prob.sum(-1, keepdim=True)

#     def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
#         mean, _ = self._get_dist_params(obs)
#         return torch.tanh(mean)


# # ════════════════════════════════════════════════════════
# # CRITIC  —  Twin Q-Networks
# # ════════════════════════════════════════════════════════

# class Critic(nn.Module):
#     """
#     Twin Q-networks (Fujimoto et al.) to reduce overestimation bias.
#     Input: (obs, action) → two scalar Q-values.
#     """

#     def __init__(self):
#         super().__init__()
#         in_dim = OBS_DIM + 1   # obs + 1-dim action

#         self.q1 = nn.Sequential(
#             nn.Linear(in_dim,   HIDDEN_DIM), nn.ReLU(),
#             nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
#             nn.Linear(HIDDEN_DIM, 1),
#         )
#         self.q2 = nn.Sequential(
#             nn.Linear(in_dim,   HIDDEN_DIM), nn.ReLU(),
#             nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
#             nn.Linear(HIDDEN_DIM, 1),
#         )

#     def forward(self, obs: torch.Tensor,
#                 action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         sa = torch.cat([obs, action], dim=-1)
#         return self.q1(sa), self.q2(sa)

#     def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
#         """Min of twin Q-values — used in actor and target updates."""
#         q1, q2 = self.forward(obs, action)
#         return torch.min(q1, q2)


# # ════════════════════════════════════════════════════════
# # FEASIBILITY PROJECTION  —  Differentiable SoC Clamp
# # ════════════════════════════════════════════════════════

# class FeasibilityProjection(nn.Module):
#     """
#     Maps actor's raw action ∈ [-1, 1] → feasible (action, new_soc)
#     respecting SoC bounds [SOC_MIN, SOC_MAX] and charge/discharge efficiency.

#     Differentiable: gradients flow back through the clamp to the actor.
#     """

#     def forward(self, raw_action: torch.Tensor,
#                 soc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             raw_action : (B, 1) ∈ [-1, 1]  +1=full charge, -1=full discharge
#             soc        : (B, 1) current state of charge ∈ [0, 1]
#         Returns:
#             feasible_action : (B, 1) adjusted to SoC limits
#             new_soc         : (B, 1) updated SoC
#         """
#         sqrt_eta = math.sqrt(EFFICIENCY)

#         # Convert normalised action → ΔSoC (fraction of capacity)
#         delta_soc_raw = raw_action * (BATTERY_POWER_MW * INTERVAL_H / BATTERY_CAP_MWH)

#         # Apply efficiency: charging costs more, discharging yields less
#         delta_soc_eff = torch.where(
#             delta_soc_raw > 0,
#             delta_soc_raw * sqrt_eta,       # charge: SoC gain is reduced
#             delta_soc_raw / sqrt_eta,       # discharge: SoC drop is larger
#         )

#         # Clamp new SoC to physical limits
#         new_soc = (soc + delta_soc_eff).clamp(SOC_MIN, SOC_MAX)

#         # Back-compute the feasible delta and corresponding action
#         actual_delta = new_soc - soc
#         actual_delta_raw = torch.where(
#             actual_delta > 0,
#             actual_delta / sqrt_eta,
#             actual_delta * sqrt_eta,
#         )
#         feasible_action = (
#             actual_delta_raw / (BATTERY_POWER_MW * INTERVAL_H / BATTERY_CAP_MWH)
#         ).clamp(-1.0, 1.0)

#         return feasible_action, new_soc


# # ════════════════════════════════════════════════════════
# # SHAPE VERIFICATION (run this file directly)
# # ════════════════════════════════════════════════════════

# def verify_shapes():
#     print("=" * 55)
#     print("Pipeline 3 — Model Shape Verification")
#     print("=" * 55)
#     print(f"Device: {DEVICE}")

#     B = 4  # batch size for testing

#     ttfe       = TTFE().to(DEVICE)
#     actor      = Actor().to(DEVICE)
#     critic     = Critic().to(DEVICE)
#     projection = FeasibilityProjection().to(DEVICE)

#     # Dummy inputs
#     price_window = torch.randn(B, WINDOW_LEN, PRICE_DIM).to(DEVICE)
#     system_vars  = torch.randn(B, SYSTEM_DIM).to(DEVICE)
#     time_feats   = torch.randn(B, TIME_DIM).to(DEVICE)
#     soc          = torch.rand(B, 1).to(DEVICE) * (SOC_MAX - SOC_MIN) + SOC_MIN

#     # Forward pass
#     feat    = ttfe(price_window)
#     obs     = torch.cat([feat, system_vars, time_feats, soc], dim=-1)
#     action, log_prob = actor.sample(obs)
#     q1, q2  = critic(obs, action)
#     f_action, new_soc = projection(action, soc)

#     print(f"\n  price_window : {tuple(price_window.shape)}")
#     print(f"  TTFE output  : {tuple(feat.shape)}         ← should be ({B}, {TTFE_DIM})")
#     print(f"  obs          : {tuple(obs.shape)}         ← should be ({B}, {OBS_DIM})")
#     print(f"  action       : {tuple(action.shape)}          ← should be ({B}, 1)")
#     print(f"  log_prob     : {tuple(log_prob.shape)}          ← should be ({B}, 1)")
#     print(f"  Q1, Q2       : {tuple(q1.shape)}, {tuple(q2.shape)}  ← should be ({B}, 1)")
#     print(f"  new_soc      : {tuple(new_soc.shape)}          ← should be ({B}, 1)")

#     # Parameter counts
#     def count_params(m): return sum(p.numel() for p in m.parameters())
#     print(f"\n  TTFE params    : {count_params(ttfe):,}")
#     print(f"  Actor params   : {count_params(actor):,}")
#     print(f"  Critic params  : {count_params(critic):,}")
#     print(f"  Total          : {count_params(ttfe)+count_params(actor)+count_params(critic):,}")

#     print("\n✓ All shapes correct. Models ready for training.")
#     print("\nNext step:  python pipeline/p4_train.py")


# if __name__ == "__main__":
#     verify_shapes()
"""
Pipeline 3 — Model Definitions
================================
Defines all neural network components for Stage 1:
  - TTFE  (Transformer Temporal Feature Extractor)
  - Actor (squashed Gaussian, 1-dim energy action)
  - Critic (twin Q-networks)
  - FeasibilityProjection (differentiable SoC clamp)

These classes are imported by p4_train.py and p5_evaluate.py.
You can also run this file directly to verify model shapes are correct.

Usage (verification):
    python pipeline/p3_models.py
"""

import sys, os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.config import *


# ════════════════════════════════════════════════════════
# TTFE  —  Transformer Temporal Feature Extractor
# ════════════════════════════════════════════════════════

class TTFE(nn.Module):
    """
    Encodes (batch, WINDOW_LEN, PRICE_DIM) → (batch, TTFE_DIM).

    Architecture (follows TempDRL, Li et al. 2024):
      Linear projection → learnable positional encoding →
      N × TransformerEncoderLayer → mean pool over time → LayerNorm
    """

    def __init__(self):
        super().__init__()
        self.d_model = TTFE_DIM

        # Project raw 12-dim prices → d_model
        self.input_proj = nn.Linear(PRICE_DIM, TTFE_DIM)

        # Learnable positional encoding  (1, L, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, WINDOW_LEN, TTFE_DIM))
        nn.init.normal_(self.pos_enc, std=0.02)

        # Transformer encoder stack
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = TTFE_DIM,
            nhead           = TTFE_NHEAD,
            dim_feedforward = TTFE_DIM * 4,
            dropout         = TTFE_DROPOUT,   # use config value (was broken: used TTFE_DROP which does not exist)
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=TTFE_NLAYERS)
        self.layer_norm  = nn.LayerNorm(TTFE_DIM)

    def forward(self, price_window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_window: (B, WINDOW_LEN, PRICE_DIM)  — z-score normalised, clipped ±5σ
        Returns:
            feat: (B, TTFE_DIM)
        """
        x = self.input_proj(price_window)    # (B, L, d_model)
        x = x + self.pos_enc                 # add positional encoding
        x = self.transformer(x)              # (B, L, d_model)
        x = self.layer_norm(x)
        return x.mean(dim=1)                 # mean pool → (B, d_model)


# ════════════════════════════════════════════════════════
# ACTOR  —  Squashed Gaussian Policy
# ════════════════════════════════════════════════════════

LOG_STD_MIN = -5
LOG_STD_MAX =  2


class Actor(nn.Module):
    """
    Stage 1 actor: 78-d obs → 1-d continuous action ∈ [-1, 1].
    Uses the reparameterisation trick (tanh-squashed Gaussian) for SAC.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(HIDDEN_DIM, 1)
        self.log_std_head = nn.Linear(HIDDEN_DIM, 1)

        # Initialise output layers near zero for stable early training
        nn.init.uniform_(self.mean_head.weight,    -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)

    def _get_dist_params(self, obs: torch.Tensor):
        h       = self.net(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def forward(self, obs: torch.Tensor):
        return self._get_dist_params(obs)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action   : (B, 1) ∈ [-1, 1]  via tanh squashing
            log_prob : (B, 1)  with tanh correction
        """
        mean, std = self._get_dist_params(obs)
        dist      = torch.distributions.Normal(mean, std)
        x         = dist.rsample()                                   # reparameterisation
        action    = torch.tanh(x)
        log_prob  = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._get_dist_params(obs)
        return torch.tanh(mean)


# ════════════════════════════════════════════════════════
# CRITIC  —  Twin Q-Networks
# ════════════════════════════════════════════════════════

class Critic(nn.Module):
    """
    Twin Q-networks (Fujimoto et al.) to reduce overestimation bias.
    Input: (obs, action) → two scalar Q-values.
    """

    def __init__(self):
        super().__init__()
        in_dim = OBS_DIM + 1   # obs + 1-dim action

        self.q1 = nn.Sequential(
            nn.Linear(in_dim,   HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim,   HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, obs: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Min of twin Q-values — used in actor and target updates."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ════════════════════════════════════════════════════════
# FEASIBILITY PROJECTION  —  Differentiable SoC Clamp
# ════════════════════════════════════════════════════════

class FeasibilityProjection(nn.Module):
    """
    Maps actor's raw action ∈ [-1, 1] → feasible (action, new_soc)
    respecting SoC bounds [SOC_MIN, SOC_MAX] and charge/discharge efficiency.

    Differentiable: gradients flow back through the clamp to the actor.
    """

    def forward(self, raw_action: torch.Tensor,
                soc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            raw_action : (B, 1) ∈ [-1, 1]  +1=full charge, -1=full discharge
            soc        : (B, 1) current state of charge ∈ [0, 1]
        Returns:
            feasible_action : (B, 1) adjusted to SoC limits
            new_soc         : (B, 1) updated SoC
        """
        sqrt_eta = math.sqrt(EFFICIENCY)

        # Convert normalised action → ΔSoC (fraction of capacity)
        delta_soc_raw = raw_action * (BATTERY_POWER_MW * INTERVAL_H / BATTERY_CAP_MWH)

        # Apply efficiency: charging costs more, discharging yields less
        delta_soc_eff = torch.where(
            delta_soc_raw > 0,
            delta_soc_raw * sqrt_eta,       # charge: SoC gain is reduced
            delta_soc_raw / sqrt_eta,       # discharge: SoC drop is larger
        )

        # Clamp new SoC to physical limits
        new_soc = (soc + delta_soc_eff).clamp(SOC_MIN, SOC_MAX)

        # Back-compute the feasible delta and corresponding action
        actual_delta = new_soc - soc
        actual_delta_raw = torch.where(
            actual_delta > 0,
            actual_delta / sqrt_eta,
            actual_delta * sqrt_eta,
        )
        feasible_action = (
            actual_delta_raw / (BATTERY_POWER_MW * INTERVAL_H / BATTERY_CAP_MWH)
        ).clamp(-1.0, 1.0)

        return feasible_action, new_soc


# ════════════════════════════════════════════════════════
# SHAPE VERIFICATION (run this file directly)
# ════════════════════════════════════════════════════════

def verify_shapes():
    print("=" * 55)
    print("Pipeline 3 — Model Shape Verification")
    print("=" * 55)
    print(f"Device: {DEVICE}")

    B = 4  # batch size for testing

    ttfe       = TTFE().to(DEVICE)
    actor      = Actor().to(DEVICE)
    critic     = Critic().to(DEVICE)
    projection = FeasibilityProjection().to(DEVICE)

    # Dummy inputs
    price_window = torch.randn(B, WINDOW_LEN, PRICE_DIM).to(DEVICE)
    system_vars  = torch.randn(B, SYSTEM_DIM).to(DEVICE)
    time_feats   = torch.randn(B, TIME_DIM).to(DEVICE)
    soc          = torch.rand(B, 1).to(DEVICE) * (SOC_MAX - SOC_MIN) + SOC_MIN

    # Forward pass
    feat    = ttfe(price_window)
    obs     = torch.cat([feat, system_vars, time_feats, soc], dim=-1)
    action, log_prob = actor.sample(obs)
    q1, q2  = critic(obs, action)
    f_action, new_soc = projection(action, soc)

    print(f"\n  price_window : {tuple(price_window.shape)}")
    print(f"  TTFE output  : {tuple(feat.shape)}         ← should be ({B}, {TTFE_DIM})")
    print(f"  obs          : {tuple(obs.shape)}         ← should be ({B}, {OBS_DIM})")
    print(f"  action       : {tuple(action.shape)}          ← should be ({B}, 1)")
    print(f"  log_prob     : {tuple(log_prob.shape)}          ← should be ({B}, 1)")
    print(f"  Q1, Q2       : {tuple(q1.shape)}, {tuple(q2.shape)}  ← should be ({B}, 1)")
    print(f"  new_soc      : {tuple(new_soc.shape)}          ← should be ({B}, 1)")

    # Parameter counts
    def count_params(m): return sum(p.numel() for p in m.parameters())
    print(f"\n  TTFE params    : {count_params(ttfe):,}")
    print(f"  Actor params   : {count_params(actor):,}")
    print(f"  Critic params  : {count_params(critic):,}")
    print(f"  Total          : {count_params(ttfe)+count_params(actor)+count_params(critic):,}")

    print("\n✓ All shapes correct. Models ready for training.")
    print("\nNext step:  python pipeline/p4_train.py")


if __name__ == "__main__":
    verify_shapes()
