"""
config.py — Central Configuration for Stage 1
================================================
Edit this file based on the output of p1_inspect_data.py.
All other pipelines import from here.

STEP 1: Run p1_inspect_data.py
STEP 2: Copy the column names it prints into this file
STEP 3: Run the remaining pipelines in order
"""

import os
import torch

# ══════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════
DATA_ROOT      = "./data/processed"
CHECKPOINT_DIR = "./checkpoints/stage1"
LOG_DIR        = "./logs"

# ══════════════════════════════════════════════════════════
# COLUMN NAMES  ← fill these in after running p1_inspect_data.py
# ══════════════════════════════════════════════════════════

# The 12 price columns fed into the TTFE.
# Order matters: [rt_lmp, rt_mcpc×5, dam_spp, dam_as×5]
# # PLACEHOLDER — replace with actual column names from your parquets:
# PRICE_COLS = [
#     "rt_lmp",           # RT energy price
#     "rt_reg_up",        # RT regulation up
#     "rt_reg_dn",        # RT regulation down
#     "rt_rrs",           # RT responsive reserve
#     "rt_ecrs",          # RT ERCOT contingency reserve
#     "rt_nonspin",       # RT non-spinning reserve
#     "dam_spp",          # DAM energy settlement price
#     "dam_reg_up",       # DAM regulation up
#     "dam_reg_dn",       # DAM regulation down
#     "dam_rrs",          # DAM responsive reserve
#     "dam_ecrs",         # DAM ERCOT contingency reserve
#     "dam_nonspin",      # DAM non-spinning reserve
# ]
# 9-dim price vector (NOT 12 — rt_mcpc cols are all null pre-RTC+B)
PRICE_COLS = [
    "rt_lmp",
    "dam_spp",
    "dam_as_regup",
    "dam_as_regdn",
    "dam_as_rrs",
    "dam_as_ecrs",
    "dam_as_nsrs",
]

# The 7 system condition columns (NOT fed into TTFE — concatenated after)
# PLACEHOLDER — replace with actual column names:
# SYSTEM_COLS = [
#     "load_fcst",        # load forecast (MW)
#     "load_act",         # actual load (MW)
#     "wind_gen",         # wind generation (MW)
#     "solar_gen",        # solar generation (MW)
#     "ercot_ind1",       # ERCOT indicator 1
#     "ercot_ind2",       # ERCOT indicator 2
#     "ercot_ind3",       # ERCOT indicator 3
# ]
SYSTEM_COLS = [
    "total_load_mw",
    "load_forecast_mw",
    "wind_actual_mw",
    "wind_forecast_mw",
    "solar_actual_mw",
    "solar_forecast_mw",
    "net_load_mw",
]
# Name of the timestamp column in the parquets (or "__index__" if it's the index)
TIMESTAMP_COL = "__index__"   # most likely; update if p1 says otherwise

# ══════════════════════════════════════════════════════════
# DATE SPLITS
# ══════════════════════════════════════════════════════════
STAGE1_START = "2020-01-01"
STAGE1_END   = "2025-12-04"    # last pre-RTC+B date (inclusive)
VAL_START    = "2025-10-01"    # hold-out for evaluation (2 months)

# ══════════════════════════════════════════════════════════
# BATTERY PHYSICAL PARAMETERS
# ══════════════════════════════════════════════════════════
BATTERY_CAP_MWH  = 100.0       # total energy capacity (MWh)
BATTERY_POWER_MW = 25.0        # max charge/discharge rate (MW)
EFFICIENCY       = 0.92        # round-trip efficiency (applied as √η per direction)
SOC_MIN          = 0.05        # min state-of-charge (fraction)
SOC_MAX          = 0.95        # max state-of-charge (fraction)
INTERVAL_H       = 5 / 60      # 5-minute intervals in hours

# ══════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════
WINDOW_LEN   = 32              # timesteps in TTFE input
# PRICE_DIM    = len(PRICE_COLS) # = 12
PRICE_DIM = len(PRICE_COLS)
SYSTEM_DIM   = len(SYSTEM_COLS)# = 7
TIME_DIM     = 6               # sin/cos time features
SOC_DIM      = 1
TTFE_DIM     = 64              # TTFE output dimension
OBS_DIM      = TTFE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM  # = 78

TTFE_NHEAD      = 4
TTFE_NLAYERS    = 2
TTFE_DROPOUT    = 0.1
HIDDEN_DIM      = 256          # SAC actor/critic hidden size
CLIP_SIGMA      = 5.0          # ±5σ clip in normalisation (handles spikes)

# ══════════════════════════════════════════════════════════
# SAC TRAINING HYPERPARAMETERS
# ══════════════════════════════════════════════════════════
REPLAY_SIZE     = 1_000_000
BATCH_SIZE      = 256
LR_ACTOR        = 3e-4
LR_CRITIC       = 3e-4
LR_ALPHA        = 3e-4
GAMMA           = 0.99
TAU             = 0.005
TARGET_ENTROPY  = -1.0         # for 1-dim action (energy only)
WARMUP_STEPS    = 5_000        # random exploration before gradient updates
TOTAL_STEPS     = 500_000
LOG_EVERY       = 1_000
SAVE_EVERY      = 50_000
EVAL_EVERY      = 10_000

# ══════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════
# assert len(PRICE_COLS)  == 12, f"Need exactly 12 price cols, got {len(PRICE_COLS)}"
assert len(PRICE_COLS) == PRICE_DIM
assert len(SYSTEM_COLS) == 7,  f"Need exactly 7 system cols, got {len(SYSTEM_COLS)}"
assert OBS_DIM == 78,          f"OBS_DIM should be 78, got {OBS_DIM}"

if __name__ == "__main__":
    print("Config loaded successfully.")
    print(f"  PRICE_COLS  ({PRICE_DIM}): {PRICE_COLS}")
    print(f"  SYSTEM_COLS ({SYSTEM_DIM}): {SYSTEM_COLS}")
    print(f"  OBS_DIM     : {OBS_DIM}")
    print(f"  DEVICE      : {DEVICE}")
