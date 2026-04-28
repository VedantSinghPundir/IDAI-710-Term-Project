# # # # # """
# # # # # config.py — Central Configuration for Stage 1
# # # # # ================================================
# # # # # Edit this file based on the output of p1_inspect_data.py.
# # # # # All other pipelines import from here.

# # # # # STEP 1: Run p1_inspect_data.py
# # # # # STEP 2: Copy the column names it prints into this file
# # # # # STEP 3: Run the remaining pipelines in order
# # # # # """

# # # # # import os
# # # # # import torch

# # # # # # ══════════════════════════════════════════════════════════
# # # # # # PATHS
# # # # # # ══════════════════════════════════════════════════════════
# # # # # DATA_ROOT      = "./data/processed"
# # # # # CHECKPOINT_DIR = "./checkpoints/stage1"
# # # # # LOG_DIR        = "./logs"

# # # # # # ══════════════════════════════════════════════════════════
# # # # # # COLUMN NAMES  ← fill these in after running p1_inspect_data.py
# # # # # # ══════════════════════════════════════════════════════════

# # # # # # The 12 price columns fed into the TTFE.
# # # # # # Order matters: [rt_lmp, rt_mcpc×5, dam_spp, dam_as×5]
# # # # # # # PLACEHOLDER — replace with actual column names from your parquets:
# # # # # # PRICE_COLS = [
# # # # # #     "rt_lmp",           # RT energy price
# # # # # #     "rt_reg_up",        # RT regulation up
# # # # # #     "rt_reg_dn",        # RT regulation down
# # # # # #     "rt_rrs",           # RT responsive reserve
# # # # # #     "rt_ecrs",          # RT ERCOT contingency reserve
# # # # # #     "rt_nonspin",       # RT non-spinning reserve
# # # # # #     "dam_spp",          # DAM energy settlement price
# # # # # #     "dam_reg_up",       # DAM regulation up
# # # # # #     "dam_reg_dn",       # DAM regulation down
# # # # # #     "dam_rrs",          # DAM responsive reserve
# # # # # #     "dam_ecrs",         # DAM ERCOT contingency reserve
# # # # # #     "dam_nonspin",      # DAM non-spinning reserve
# # # # # # ]
# # # # # # 9-dim price vector (NOT 12 — rt_mcpc cols are all null pre-RTC+B)
# # # # # PRICE_COLS = [
# # # # #     "rt_lmp",
# # # # #     "dam_spp",
# # # # #     "dam_as_regup",
# # # # #     "dam_as_regdn",
# # # # #     "dam_as_rrs",
# # # # #     "dam_as_ecrs",
# # # # #     "dam_as_nsrs",
# # # # # ]

# # # # # # The 7 system condition columns (NOT fed into TTFE — concatenated after)
# # # # # # PLACEHOLDER — replace with actual column names:
# # # # # # SYSTEM_COLS = [
# # # # # #     "load_fcst",        # load forecast (MW)
# # # # # #     "load_act",         # actual load (MW)
# # # # # #     "wind_gen",         # wind generation (MW)
# # # # # #     "solar_gen",        # solar generation (MW)
# # # # # #     "ercot_ind1",       # ERCOT indicator 1
# # # # # #     "ercot_ind2",       # ERCOT indicator 2
# # # # # #     "ercot_ind3",       # ERCOT indicator 3
# # # # # # ]
# # # # # SYSTEM_COLS = [
# # # # #     "total_load_mw",
# # # # #     "load_forecast_mw",
# # # # #     "wind_actual_mw",
# # # # #     "wind_forecast_mw",
# # # # #     "solar_actual_mw",
# # # # #     "solar_forecast_mw",
# # # # #     "net_load_mw",
# # # # # ]
# # # # # # Name of the timestamp column in the parquets (or "__index__" if it's the index)
# # # # # TIMESTAMP_COL = "__index__"   # most likely; update if p1 says otherwise

# # # # # # ══════════════════════════════════════════════════════════
# # # # # # DATE SPLITS
# # # # # # ══════════════════════════════════════════════════════════
# # # # # STAGE1_START = "2020-01-01"
# # # # # STAGE1_END   = "2025-12-04"    # last pre-RTC+B date (inclusive)
# # # # # VAL_START    = "2025-10-01"    # hold-out for evaluation (2 months)

# # # # # # ══════════════════════════════════════════════════════════
# # # # # # BATTERY PHYSICAL PARAMETERS
# # # # # # ══════════════════════════════════════════════════════════
# # # # # BATTERY_CAP_MWH  = 100.0       # total energy capacity (MWh)
# # # # # BATTERY_POWER_MW = 25.0        # max charge/discharge rate (MW)
# # # # # EFFICIENCY       = 0.92        # round-trip efficiency (applied as √η per direction)
# # # # # SOC_MIN          = 0.05        # min state-of-charge (fraction)
# # # # # SOC_MAX          = 0.95        # max state-of-charge (fraction)
# # # # # INTERVAL_H       = 5 / 60      # 5-minute intervals in hours

# # # # # # ══════════════════════════════════════════════════════════
# # # # # # MODEL ARCHITECTURE
# # # # # # ══════════════════════════════════════════════════════════
# # # # # # WINDOW_LEN   = 32              # timesteps in TTFE input
# # # # # WINDOW_LEN  = 288 
# # # # # # PRICE_DIM    = len(PRICE_COLS) # = 12
# # # # # PRICE_DIM = len(PRICE_COLS)
# # # # # SYSTEM_DIM   = len(SYSTEM_COLS)# = 7
# # # # # TIME_DIM     = 6               # sin/cos time features
# # # # # SOC_DIM      = 1
# # # # # TTFE_DIM     = 64              # TTFE output dimension
# # # # # OBS_DIM      = TTFE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM  # = 78

# # # # # TTFE_NHEAD      = 4
# # # # # TTFE_NLAYERS    = 2
# # # # # TTFE_DROPOUT    = 0.1
# # # # # HIDDEN_DIM      = 256          # SAC actor/critic hidden size
# # # # # CLIP_SIGMA      = 5.0          # ±5σ clip in normalisation (handles spikes)

# # # # # # ══════════════════════════════════════════════════════════
# # # # # # SAC TRAINING HYPERPARAMETERS
# # # # # # ══════════════════════════════════════════════════════════
# # # # # REPLAY_SIZE     = 1_000_000
# # # # # BATCH_SIZE      = 256
# # # # # LR_ACTOR        = 3e-4
# # # # # LR_CRITIC       = 3e-4
# # # # # LR_ALPHA        = 3e-4
# # # # # GAMMA           = 0.99
# # # # # TAU             = 0.005
# # # # # TARGET_ENTROPY  = -0.5         # for 1-dim action (energy only)
# # # # # # TARGET_ENTROPY = -0.1 #Fix: lower TARGET_ENTROPY to -0.1 to stop alpha explosion
# # # # # WARMUP_STEPS    = 5_000        # random exploration before gradient updates
# # # # # # TOTAL_STEPS     = 500_000
# # # # # TOTAL_STEPS = 50000
# # # # # LOG_EVERY       = 1_000
# # # # # SAVE_EVERY      = 50_000
# # # # # EVAL_EVERY      = 10_000
# # # # # REWARD_SCALE    = 100.0        # fixed reward divisor (÷100 keeps Q-values stable)
# # # # # MAX_EP_STEPS    = 288          # one trading day = 288 five-minute intervals
# # # # # # ══════════════════════════════════════════════════════════
# # # # # # DEVICE
# # # # # # ══════════════════════════════════════════════════════════
# # # # # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # # # # ══════════════════════════════════════════════════════════
# # # # # # SANITY CHECK
# # # # # # ══════════════════════════════════════════════════════════
# # # # # # assert len(PRICE_COLS)  == 12, f"Need exactly 12 price cols, got {len(PRICE_COLS)}"
# # # # # assert len(PRICE_COLS) == PRICE_DIM
# # # # # assert len(SYSTEM_COLS) == 7,  f"Need exactly 7 system cols, got {len(SYSTEM_COLS)}"
# # # # # assert OBS_DIM == 78,          f"OBS_DIM should be 78, got {OBS_DIM}"

# # # # # if __name__ == "__main__":
# # # # #     print("Config loaded successfully.")
# # # # #     print(f"  PRICE_COLS  ({PRICE_DIM}): {PRICE_COLS}")
# # # # #     print(f"  SYSTEM_COLS ({SYSTEM_DIM}): {SYSTEM_COLS}")
# # # # #     print(f"  OBS_DIM     : {OBS_DIM}")
# # # # #     print(f"  DEVICE      : {DEVICE}")
# # # # """
# # # # config.py — Central Configuration for Stage 1
# # # # ================================================
# # # # Edit this file based on the output of p1_inspect_data.py.
# # # # All other pipelines import from here.

# # # # STEP 1: Run p1_inspect_data.py
# # # # STEP 2: Copy the column names it prints into this file
# # # # STEP 3: Run the remaining pipelines in order

# # # # IMPORTANT — BEFORE RUNNING p4_train.py:
# # # #   If STAGE1_START has changed (now 2022-01-01), you MUST rerun
# # # #   p2_build_dataset.py to recompute normaliser_stats.npz from the
# # # #   new date range. Using old stats from 2020-2025 on 2022-2025 data
# # # #   will produce incorrect z-score normalisation.
# # # # """

# # # # import os
# # # # import torch

# # # # # ══════════════════════════════════════════════════════════
# # # # # PATHS
# # # # # ══════════════════════════════════════════════════════════
# # # # DATA_ROOT      = "./data/processed"
# # # # CHECKPOINT_DIR = "./checkpoints/stage1"
# # # # LOG_DIR        = "./logs"

# # # # # ══════════════════════════════════════════════════════════
# # # # # COLUMN NAMES
# # # # # ══════════════════════════════════════════════════════════

# # # # # 7-dim price vector (NOT 12 — rt_mcpc cols are all null pre-RTC+B)
# # # # PRICE_COLS = [
# # # #     "rt_lmp",
# # # #     "dam_spp",
# # # #     "dam_as_regup",
# # # #     "dam_as_regdn",
# # # #     "dam_as_rrs",
# # # #     "dam_as_ecrs",
# # # #     "dam_as_nsrs",
# # # # ]

# # # # # 7 system condition columns (concatenated after TTFE, not fed into it)
# # # # SYSTEM_COLS = [
# # # #     "total_load_mw",
# # # #     "load_forecast_mw",
# # # #     "wind_actual_mw",
# # # #     "wind_forecast_mw",
# # # #     "solar_actual_mw",
# # # #     "solar_forecast_mw",
# # # #     "net_load_mw",
# # # # ]

# # # # TIMESTAMP_COL = "__index__"

# # # # # ══════════════════════════════════════════════════════════
# # # # # DATE SPLITS
# # # # # ══════════════════════════════════════════════════════════
# # # # # Changed from 2020-01-01 to 2022-01-01 to:
# # # # #   - Exclude 2021 Winter Storm Uri outlier (kurtosis 786 → 631)
# # # # #   - Use more stationary market regime (year-to-year std variation 12x → 3x)
# # # # #   - NOTE: kurtosis is still ~631 — cannot be eliminated by data selection
# # # # #     Huber loss and improved training are STILL necessary
# # # # STAGE1_START = "2022-01-01"
# # # # STAGE1_END   = "2025-12-04"
# # # # VAL_START    = "2025-10-01"

# # # # # ══════════════════════════════════════════════════════════
# # # # # BATTERY PHYSICAL PARAMETERS
# # # # # ══════════════════════════════════════════════════════════
# # # # BATTERY_CAP_MWH  = 100.0
# # # # BATTERY_POWER_MW = 25.0
# # # # EFFICIENCY       = 0.92
# # # # SOC_MIN          = 0.05
# # # # SOC_MAX          = 0.95
# # # # INTERVAL_H       = 5 / 60

# # # # # ══════════════════════════════════════════════════════════
# # # # # MODEL ARCHITECTURE
# # # # # ══════════════════════════════════════════════════════════
# # # # # WINDOW_LEN=288 was tested and failed (5x worse, critic 4x less stable).
# # # # # Root cause unclear (TTFE gradient scaling or demo coverage).
# # # # # Keeping at 32 until root cause is understood.
# # # # WINDOW_LEN   = 32
# # # # PRICE_DIM    = len(PRICE_COLS)   # = 7
# # # # SYSTEM_DIM   = len(SYSTEM_COLS)  # = 7
# # # # TIME_DIM     = 6
# # # # SOC_DIM      = 1
# # # # TTFE_DIM     = 64
# # # # OBS_DIM      = TTFE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM  # = 78

# # # # TTFE_NHEAD      = 4
# # # # TTFE_NLAYERS    = 2
# # # # TTFE_DROPOUT    = 0.1
# # # # HIDDEN_DIM      = 256
# # # # CLIP_SIGMA      = 5.0

# # # # # ══════════════════════════════════════════════════════════
# # # # # SAC TRAINING HYPERPARAMETERS
# # # # # ══════════════════════════════════════════════════════════

# # # # # --- Replay buffers (two separate buffers) ---
# # # # DEMO_BUFFER_SIZE  = 100_000      # demo transitions only
# # # # AGENT_BUFFER_SIZE = 1_000_000    # agent-generated transitions

# # # # # --- Demonstration settings ---
# # # # DEMO_STEPS        = 50_000       # rule-based demo transitions to collect
# # # #                                   # 50k ÷ 288 steps/episode ≈ 173 diverse episodes
# # # #                                   # (vs 34 previously with 10k — 5x more coverage)
# # # # DEMO_FLOOR        = 0.05         # minimum demo sampling ratio — NEVER goes to 0
# # # #                                   # keeps charge/discharge diversity throughout training
# # # # DEMO_DECAY_STEPS  = 200_000      # steps over which demo ratio decays from 1.0 to DEMO_FLOOR
# # # #                                   # longer decay than before (was 50k) because policy
# # # #                                   # collapsed within 10-20k steps in previous run

# # # # # --- Critic stability (key fixes for kurtosis 786 problem) ---
# # # # HUBER_DELTA       = 10.0         # Huber loss threshold
# # # #                                   # Errors < 10 → quadratic (normal intervals)
# # # #                                   # Errors > 10 → linear (spike intervals capped)
# # # #                                   # Reduces spike gradient dominance from 40,580x to ~15x
# # # #                                   # Placement: Q-target mean=6.93, max=77.26 → delta=10 is correct

# # # # # --- Learning rates ---
# # # # LR_ACTOR          = 3e-4
# # # # LR_CRITIC         = 1e-4         # Lowered from 3e-4 — training log showed critic
# # # #                                   # climbing steadily before explosion; slower updates
# # # #                                   # give target network time to stabilise
# # # # LR_ALPHA          = 3e-4

# # # # # --- Gradient clipping ---
# # # # GRAD_CLIP         = 0.5          # Tightened from 1.0 — prevents large gradient
# # # #                                   # updates during spike transitions

# # # # # --- SAC core ---
# # # # BATCH_SIZE        = 256
# # # # GAMMA             = 0.99
# # # # TAU               = 0.005
# # # # TARGET_ENTROPY    = -0.5         # for 1-dim action space

# # # # # --- Episode and training length ---
# # # # MAX_EP_STEPS      = 288          # one trading day
# # # # TOTAL_STEPS       = 500_000

# # # # # --- Logging and saving ---
# # # # LOG_EVERY         = 1_000
# # # # SAVE_EVERY        = 50_000
# # # # EVAL_EVERY        = 10_000

# # # # # --- Reward scaling ---
# # # # REWARD_SCALE      = 100.0        # fixed — proven correct range [-4.5, +188]

# # # # # ══════════════════════════════════════════════════════════
# # # # # EARLY STOPPING AND HEALTH CRITERIA
# # # # # ══════════════════════════════════════════════════════════
# # # # # Based on diagnostics from Stage 1 Plan A training:
# # # # #   - Critic exploded at step 380k (loss went from ~50 to ~1000)
# # # # #   - Policy collapsed (log_pi = +0.97, should be in [-2, 0])
# # # # # These thresholds trigger automatic stopping before wasting compute.

# # # # CRITIC_LOSS_STOP      = 300      # stop if 100-step MA of critic loss > 300
# # # #                                   # (training log: critic was ~50 healthy, ~1000 exploded)
# # # # LOG_PI_STOP           = 0.0      # stop if 100-step MA of log_pi > 0.0
# # # #                                   # log_pi > 0 is theoretically impossible for healthy policy
# # # #                                   # (measured +0.97 in collapsed policy)
# # # # MIN_STEP_BEFORE_STOP  = 50_000   # don't trigger early stop before this step
# # # #                                   # (early training can be temporarily noisy)
# # # # CHARGE_FRAC_MIN       = 0.05     # warn if charge fraction < 5% over last 1000 steps
# # # #                                   # (indicates always-discharge collapse)

# # # # # ══════════════════════════════════════════════════════════
# # # # # DEVICE
# # # # # ══════════════════════════════════════════════════════════
# # # # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # # # ══════════════════════════════════════════════════════════
# # # # # SANITY CHECK
# # # # # ══════════════════════════════════════════════════════════
# # # # assert len(PRICE_COLS)  == PRICE_DIM
# # # # assert len(SYSTEM_COLS) == 7,  f"Need 7 system cols, got {len(SYSTEM_COLS)}"
# # # # assert OBS_DIM          == 78, f"OBS_DIM should be 78, got {OBS_DIM}"
# # # # assert WINDOW_LEN       == 32, f"WINDOW_LEN should be 32, got {WINDOW_LEN}"
# # # # assert DEMO_FLOOR       >  0,  "DEMO_FLOOR must be > 0 — never let demo ratio reach 0"
# # # # assert HUBER_DELTA      >  0,  "HUBER_DELTA must be positive"

# # # # if __name__ == "__main__":
# # # #     print("Config loaded successfully.")
# # # #     print(f"  STAGE1_START: {STAGE1_START}  ← excludes 2021 Winter Storm Uri")
# # # #     print(f"  PRICE_COLS  ({PRICE_DIM}): {PRICE_COLS}")
# # # #     print(f"  SYSTEM_COLS ({SYSTEM_DIM}): {SYSTEM_COLS}")
# # # #     print(f"  OBS_DIM     : {OBS_DIM}")
# # # #     print(f"  WINDOW_LEN  : {WINDOW_LEN}")
# # # #     print(f"  DEVICE      : {DEVICE}")
# # # #     print(f"  LR_CRITIC   : {LR_CRITIC}  (lowered from 3e-4)")
# # # #     print(f"  GRAD_CLIP   : {GRAD_CLIP}  (tightened from 1.0)")
# # # #     print(f"  HUBER_DELTA : {HUBER_DELTA}")
# # # #     print(f"  DEMO_STEPS  : {DEMO_STEPS:,}")
# # # #     print(f"  DEMO_FLOOR  : {DEMO_FLOOR}  (never decays to 0)")
# # # #     print()
# # # #     print("  REMINDER: Rerun p2_build_dataset.py before training!")
# # # #     print("  (normaliser_stats.npz must be recomputed for 2022-2025 data)")
# # # """
# # # config.py — Central Configuration for Stage 1
# # # ================================================
# # # Edit this file based on the output of p1_inspect_data.py.
# # # All other pipelines import from here.

# # # STEP 1: Run p1_inspect_data.py
# # # STEP 2: Copy the column names it prints into this file
# # # STEP 3: Run the remaining pipelines in order

# # # IMPORTANT — BEFORE RUNNING p4_train.py:
# # #   If STAGE1_START has changed (now 2022-01-01), you MUST rerun
# # #   p2_build_dataset.py to recompute normaliser_stats.npz from the
# # #   new date range. Using old stats from 2020-2025 on 2022-2025 data
# # #   will produce incorrect z-score normalisation.
# # # """

# # # import os
# # # import torch

# # # # ══════════════════════════════════════════════════════════
# # # # PATHS
# # # # ══════════════════════════════════════════════════════════
# # # DATA_ROOT      = "./data/processed"
# # # CHECKPOINT_DIR = "./checkpoints/stage1"
# # # LOG_DIR        = "./logs"

# # # # ══════════════════════════════════════════════════════════
# # # # COLUMN NAMES
# # # # ══════════════════════════════════════════════════════════

# # # # 7-dim price vector (NOT 12 — rt_mcpc cols are all null pre-RTC+B)
# # # PRICE_COLS = [
# # #     "rt_lmp",
# # #     "dam_spp",
# # #     "dam_as_regup",
# # #     "dam_as_regdn",
# # #     "dam_as_rrs",
# # #     "dam_as_ecrs",
# # #     "dam_as_nsrs",
# # # ]

# # # # 7 system condition columns (concatenated after TTFE, not fed into it)
# # # SYSTEM_COLS = [
# # #     "total_load_mw",
# # #     "load_forecast_mw",
# # #     "wind_actual_mw",
# # #     "wind_forecast_mw",
# # #     "solar_actual_mw",
# # #     "solar_forecast_mw",
# # #     "net_load_mw",
# # # ]

# # # TIMESTAMP_COL = "__index__"

# # # # ══════════════════════════════════════════════════════════
# # # # DATE SPLITS
# # # # ══════════════════════════════════════════════════════════
# # # # Changed from 2020-01-01 to 2022-01-01 to:
# # # #   - Exclude 2021 Winter Storm Uri outlier (kurtosis 786 → 631)
# # # #   - Use more stationary market regime (year-to-year std variation 12x → 3x)
# # # #   - NOTE: kurtosis is still ~631 — Huber loss is still necessary
# # # STAGE1_START = "2022-01-01"
# # # STAGE1_END   = "2025-12-04"
# # # VAL_START    = "2025-10-01"

# # # # ══════════════════════════════════════════════════════════
# # # # BATTERY PHYSICAL PARAMETERS
# # # # ══════════════════════════════════════════════════════════
# # # BATTERY_CAP_MWH  = 100.0
# # # BATTERY_POWER_MW = 25.0
# # # EFFICIENCY       = 0.92
# # # SOC_MIN          = 0.05
# # # SOC_MAX          = 0.95
# # # INTERVAL_H       = 5 / 60

# # # # ══════════════════════════════════════════════════════════
# # # # MODEL ARCHITECTURE
# # # # ══════════════════════════════════════════════════════════
# # # # WINDOW_LEN=288 tested and failed (5x worse revenue, 4x worse critic).
# # # # Keeping at 32 until reward shaping is confirmed working.
# # # # Sequence: reward shaping first → then test 288 → then DAM features.
# # # WINDOW_LEN   = 32
# # # PRICE_DIM    = len(PRICE_COLS)   # = 7
# # # SYSTEM_DIM   = len(SYSTEM_COLS)  # = 7
# # # TIME_DIM     = 6
# # # SOC_DIM      = 1
# # # TTFE_DIM     = 64
# # # OBS_DIM      = TTFE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM  # = 78

# # # TTFE_NHEAD      = 4
# # # TTFE_NLAYERS    = 2
# # # TTFE_DROPOUT    = 0.1
# # # HIDDEN_DIM      = 256
# # # CLIP_SIGMA      = 5.0

# # # # ══════════════════════════════════════════════════════════
# # # # SAC TRAINING HYPERPARAMETERS
# # # # ══════════════════════════════════════════════════════════

# # # # --- Replay buffers (two separate buffers) ---
# # # DEMO_BUFFER_SIZE  = 100_000
# # # AGENT_BUFFER_SIZE = 1_000_000

# # # # --- Demonstration settings ---
# # # DEMO_STEPS        = 50_000       # ~173 diverse 288-step episodes
# # # DEMO_FLOOR        = 0.05         # never decays to 0 — preserves charge/discharge diversity
# # # DEMO_DECAY_STEPS  = 200_000      # slow decay so policy has time to learn before demos fade

# # # # --- Reward design ---
# # # # Inventory-adjusted reward formula (potential-based shaping):
# # # #   grid_mwh = -action × BATTERY_POWER_MW × INTERVAL_H
# # # #   reward   = grid_mwh × (rt_lmp - p_ref) / REWARD_SCALE
# # # #            - CYCLE_COST_PER_MWH × |grid_mwh| / REWARD_SCALE
# # # #
# # # # Where p_ref = training median price (~$24.21 for 2022-2025 data)
# # # #
# # # # This gives:
# # # #   Low price ($9):  charge → positive reward, discharge → negative reward
# # # #   High price ($40): discharge → positive reward, charge → negative reward
# # # #   Near median:     hold → optimal (degradation cost makes small trades unprofitable)
# # # #
# # # # Theoretical basis: potential-based reward shaping (Ng et al., 1999)
# # # # Guaranteed to preserve the optimal policy of the original cash reward.
# # # #
# # # REWARD_SCALE       = 100.0        # fixed divisor to keep Q-values stable
# # # CYCLE_COST_PER_MWH = 1.0         # battery degradation cost per MWh cycled
# # #                                   # $1/MWh is conservative (real cost: $5-15/MWh)
# # #                                   # Makes holding optimal when spread < $1/MWh
# # #                                   # Increase to $3-5 if agent still trades too much

# # # # --- Critic stability ---
# # # HUBER_DELTA       = 10.0          # Reduces spike gradient dominance from 40,580x to ~15x
# # # LR_ACTOR          = 3e-4
# # # LR_CRITIC         = 1e-4          # Lowered from 3e-4 for stability
# # # LR_ALPHA          = 3e-4
# # # GRAD_CLIP         = 0.5           # Tightened from 1.0

# # # # --- SAC core ---
# # # BATCH_SIZE        = 256
# # # GAMMA             = 0.99          # Test 0.995 after reward shaping is confirmed working
# # # TAU               = 0.005
# # # TARGET_ENTROPY    = -0.5

# # # # --- Episode and training ---
# # # MAX_EP_STEPS      = 288
# # # TOTAL_STEPS       = 500_000

# # # # --- Logging and saving ---
# # # LOG_EVERY         = 1_000
# # # SAVE_EVERY        = 50_000
# # # EVAL_EVERY        = 10_000

# # # # ══════════════════════════════════════════════════════════
# # # # EARLY STOPPING AND HEALTH CRITERIA
# # # # ══════════════════════════════════════════════════════════
# # # CRITIC_LOSS_STOP      = 300
# # # LOG_PI_STOP           = 0.0
# # # MIN_STEP_BEFORE_STOP  = 50_000
# # # CHARGE_FRAC_MIN       = 0.05

# # # # ══════════════════════════════════════════════════════════
# # # # DEVICE
# # # # ══════════════════════════════════════════════════════════
# # # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # # ══════════════════════════════════════════════════════════
# # # # SANITY CHECK
# # # # ══════════════════════════════════════════════════════════
# # # assert len(PRICE_COLS)  == PRICE_DIM
# # # assert len(SYSTEM_COLS) == 7,  f"Need 7 system cols, got {len(SYSTEM_COLS)}"
# # # assert OBS_DIM          == 78, f"OBS_DIM should be 78, got {OBS_DIM}"
# # # assert WINDOW_LEN       == 32, f"WINDOW_LEN should be 32, got {WINDOW_LEN}"
# # # assert DEMO_FLOOR       >  0,  "DEMO_FLOOR must be > 0"
# # # assert HUBER_DELTA      >  0,  "HUBER_DELTA must be positive"
# # # assert CYCLE_COST_PER_MWH >= 0, "Degradation cost must be non-negative"

# # # if __name__ == "__main__":
# # #     print("Config loaded successfully.")
# # #     print(f"  STAGE1_START       : {STAGE1_START}  (excludes 2021)")
# # #     print(f"  PRICE_COLS  ({PRICE_DIM})    : {PRICE_COLS}")
# # #     print(f"  SYSTEM_COLS ({SYSTEM_DIM})   : {SYSTEM_COLS}")
# # #     print(f"  OBS_DIM            : {OBS_DIM}")
# # #     print(f"  WINDOW_LEN         : {WINDOW_LEN}")
# # #     print(f"  DEVICE             : {DEVICE}")
# # #     print(f"  LR_CRITIC          : {LR_CRITIC}  (lowered from 3e-4)")
# # #     print(f"  GRAD_CLIP          : {GRAD_CLIP}  (tightened from 1.0)")
# # #     print(f"  HUBER_DELTA        : {HUBER_DELTA}")
# # #     print(f"  CYCLE_COST_PER_MWH : {CYCLE_COST_PER_MWH}  (degradation shaping)")
# # #     print(f"  REWARD_SCALE       : {REWARD_SCALE}")
# # #     print(f"  DEMO_STEPS         : {DEMO_STEPS:,}")
# # #     print(f"  DEMO_FLOOR         : {DEMO_FLOOR}  (never decays to 0)")
# # #     print()
# # #     print("  REMINDER: p2_build_dataset.py must be run before training")
# # #     print("  if STAGE1_START changed — normaliser_stats.npz must match dataset.")

# # """
# # config.py — Central Configuration for Stage 1
# # ================================================
# # Pipeline D: EMA-based two-term reward + fixed alpha.

# # KEY CHANGES FROM PLAN C:
# #   1. ALPHA_FIXED = 0.2  — no automatic tuning (matches paper's SAC v1)
# #   2. BETA_S = 10.0      — arbitrage bonus coefficient (from paper Table I)
# #   3. TAU_S = 0.9        — EMA smoothing parameter (from paper Table I)
# #   4. CYCLE_COST removed — degradation now handled inside two-term reward

# # WHY THESE CHANGES:
# #   - Paper Table I has no η_α (no alpha learning rate) → alpha is fixed
# #   - Paper Eq 26 uses EMA-based two-term reward that DIRECTLY penalises
# #     wrong-direction trading. Our inventory reward still allowed small
# #     positive discharge reward at low prices. The bonus term eliminates this.
# #   - EMA adapts to current market regime automatically — no look-ahead bias
# #     from a global 5-year median threshold.

# # IMPORTANT: p2_build_dataset.py does NOT need to be rerun.
# #   normaliser_stats.npz from the 2022-2025 run is still valid.
# # """

# # import os
# # import torch

# # # ══════════════════════════════════════════════════════════
# # # PATHS
# # # ══════════════════════════════════════════════════════════
# # DATA_ROOT      = "./data/processed"
# # CHECKPOINT_DIR = "./checkpoints/stage1"
# # LOG_DIR        = "./logs"

# # # ══════════════════════════════════════════════════════════
# # # COLUMN NAMES
# # # ══════════════════════════════════════════════════════════
# # PRICE_COLS = [
# #     "rt_lmp",
# #     "dam_spp",
# #     "dam_as_regup",
# #     "dam_as_regdn",
# #     "dam_as_rrs",
# #     "dam_as_ecrs",
# #     "dam_as_nsrs",
# # ]
# # SYSTEM_COLS = [
# #     "total_load_mw",
# #     "load_forecast_mw",
# #     "wind_actual_mw",
# #     "wind_forecast_mw",
# #     "solar_actual_mw",
# #     "solar_forecast_mw",
# #     "net_load_mw",
# # ]
# # TIMESTAMP_COL = "__index__"

# # # ══════════════════════════════════════════════════════════
# # # DATE SPLITS
# # # ══════════════════════════════════════════════════════════
# # STAGE1_START = "2022-01-01"   # excludes 2021 Winter Storm Uri (kurtosis 786→631)
# # STAGE1_END   = "2025-12-04"
# # VAL_START    = "2025-10-01"

# # # ══════════════════════════════════════════════════════════
# # # BATTERY PHYSICAL PARAMETERS
# # # ══════════════════════════════════════════════════════════
# # BATTERY_CAP_MWH  = 100.0
# # BATTERY_POWER_MW = 25.0
# # EFFICIENCY       = 0.92
# # SOC_MIN          = 0.05
# # SOC_MAX          = 0.95
# # INTERVAL_H       = 5 / 60

# # # ══════════════════════════════════════════════════════════
# # # MODEL ARCHITECTURE
# # # ══════════════════════════════════════════════════════════
# # WINDOW_LEN   = 32
# # PRICE_DIM    = len(PRICE_COLS)   # = 7
# # SYSTEM_DIM   = len(SYSTEM_COLS)  # = 7
# # TIME_DIM     = 6
# # SOC_DIM      = 1
# # TTFE_DIM     = 64
# # OBS_DIM      = TTFE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM   # = 78

# # TTFE_NHEAD   = 4
# # TTFE_NLAYERS = 2
# # TTFE_DROPOUT = 0.1
# # HIDDEN_DIM   = 256
# # CLIP_SIGMA   = 5.0

# # # ══════════════════════════════════════════════════════════
# # # REWARD DESIGN — Two-Term EMA Reward (from paper Eq. 26)
# # # ══════════════════════════════════════════════════════════
# # #
# # # reward = (term1 + term2) / REWARD_SCALE - degradation / REWARD_SCALE
# # #
# # # term1 = grid_mwh × rt_lmp
# # #         (cash revenue: positive when selling, negative when buying)
# # #
# # # term2 = BETA_S × |grid_mwh| × |rt_lmp - ema_price| × direction_correct
# # #         (arbitrage bonus: large when trading in correct direction)
# # #         direction_correct = 1 if discharging above EMA OR charging below EMA
# # #         direction_correct = 0 if trading in wrong direction
# # #
# # # EMA (exponential moving average):
# # #   ema_t = TAU_S × ema_{t-1} + (1 - TAU_S) × rt_lmp_t
# # #   With TAU_S=0.9, effective lookback ≈ 10 steps = 50 minutes
# # #   Adapts to current market regime — no look-ahead bias
# # #
# # # Why this fixes policy collapse:
# # #   At $9/MWh with EMA=$24: charge bonus = 10 × 2.083 × $15 / 100 = $3.13
# # #                            discharge gets NO bonus (wrong direction)
# # #   → Q(charge) >> Q(discharge) at low prices. Collapse impossible.
# # #
# # REWARD_SCALE      = 100.0    # fixed divisor for Q-value stability
# # BETA_S            = 10.0     # arbitrage bonus coefficient (paper Table I)
# # TAU_S             = 0.9      # EMA smoothing parameter (paper Table I)
# # CYCLE_COST_PER_MWH= 1.0      # degradation cost per MWh (paper uses c=AU$1/MWh)

# # # ══════════════════════════════════════════════════════════
# # # SAC HYPERPARAMETERS
# # # ══════════════════════════════════════════════════════════

# # # --- Fixed alpha (no automatic tuning) ---
# # # Paper Table I has no η_α → SAC v1 with fixed temperature.
# # # Alpha=0.2 is the last value where our training showed healthy behaviour
# # # (charge%=35%, log_pi<0) before collapse in previous runs.
# # # With BETA_S=10 making reward signal strong, smaller alpha is appropriate.
# # ALPHA_FIXED   = 0.2

# # # --- Replay buffers ---
# # DEMO_BUFFER_SIZE  = 100_000
# # AGENT_BUFFER_SIZE = 1_000_000

# # # --- Demonstrations ---
# # DEMO_STEPS       = 50_000    # ~173 diverse 288-step episodes
# # DEMO_FLOOR       = 0.05      # never decays to 0 — permanent diversity floor
# # DEMO_DECAY_STEPS = 200_000   # slow decay

# # # --- Critic stability ---
# # HUBER_DELTA  = 10.0   # reduces spike gradient dominance from 40,580x to ~15x
# # LR_ACTOR     = 3e-4
# # LR_CRITIC    = 1e-4   # lowered from 3e-4
# # LR_ALPHA     = 3e-4   # kept for reference but NOT used (alpha is fixed)
# # GRAD_CLIP    = 0.5    # tightened from 1.0

# # # --- SAC core ---
# # BATCH_SIZE      = 256
# # GAMMA           = 0.99
# # TAU             = 0.005
# # TARGET_ENTROPY  = -1.0   # kept for reference but NOT used (alpha is fixed)
# # MAX_EP_STEPS    = 288    # one trading day
# # TOTAL_STEPS     = 500_000

# # # --- Logging ---
# # LOG_EVERY  = 1_000
# # SAVE_EVERY = 50_000
# # EVAL_EVERY = 10_000

# # # ══════════════════════════════════════════════════════════
# # # EARLY STOPPING CRITERIA
# # # ══════════════════════════════════════════════════════════
# # CRITIC_LOSS_STOP     = 300    # stop if critic MA > 300 after step 50k
# # LOG_PI_STOP          = 0.0    # stop if log_pi MA > 0.0 after step 20k
# # MIN_STEP_BEFORE_STOP = 50_000
# # CHARGE_FRAC_MIN      = 0.05   # warn if charge% < 5%

# # # ══════════════════════════════════════════════════════════
# # # DEVICE
# # # ══════════════════════════════════════════════════════════
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # ══════════════════════════════════════════════════════════
# # # SANITY CHECKS
# # # ══════════════════════════════════════════════════════════
# # assert len(PRICE_COLS)  == PRICE_DIM
# # assert len(SYSTEM_COLS) == 7,  f"Need 7 system cols, got {len(SYSTEM_COLS)}"
# # assert OBS_DIM          == 78, f"OBS_DIM should be 78, got {OBS_DIM}"
# # assert WINDOW_LEN       == 32
# # assert DEMO_FLOOR       >  0
# # assert HUBER_DELTA      >  0
# # assert ALPHA_FIXED      >  0
# # assert BETA_S           >  0
# # assert 0 < TAU_S        <  1

# # if __name__ == "__main__":
# #     print("Config loaded successfully.")
# #     print(f"  STAGE1_START  : {STAGE1_START}")
# #     print(f"  OBS_DIM       : {OBS_DIM}")
# #     print(f"  DEVICE        : {DEVICE}")
# #     print(f"  ALPHA_FIXED   : {ALPHA_FIXED}  (no automatic tuning)")
# #     print(f"  BETA_S        : {BETA_S}   (arbitrage bonus from paper Table I)")
# #     print(f"  TAU_S         : {TAU_S}   (EMA smoothing from paper Table I)")
# #     print(f"  HUBER_DELTA   : {HUBER_DELTA}")
# #     print(f"  LR_CRITIC     : {LR_CRITIC}")
# #     print(f"  GRAD_CLIP     : {GRAD_CLIP}")
# #     print(f"  DEMO_STEPS    : {DEMO_STEPS:,}")
# #     print(f"  DEMO_FLOOR    : {DEMO_FLOOR}")

# """
# config.py — Central Configuration for Stage 1
# ================================================
# Plan C + Fixed Alpha (clean reset after projection bug fixes).

# KEY DECISIONS:
#   - Plan C inventory-adjusted reward (Ng et al. 1999 potential-based shaping)
#   - Fixed alpha = 0.2 (auto-tuning collapsed to 0.03 in every previous run)
#   - Projection fixes: feasible actions everywhere (critic target, actor update,
#     demo buffer) — the core correctness fix in this version
#   - NO EMA reward, NO BETA_S, NO TAU_S — those are Plan D ablation material

# REWARD FORMULA:
#     grid_mwh = -action * BATTERY_POWER_MW * INTERVAL_H
#                positive = selling (discharge), negative = buying (charge)

#     reward = grid_mwh * (rt_lmp - p_ref) / REWARD_SCALE
#            - CYCLE_COST_PER_MWH * |grid_mwh| / REWARD_SCALE

#     p_ref = training median rt_lmp (~$24.21 for 2022-2025 data)

#     Effect:
#         Low price ($9):   charge → positive, discharge → negative  ✓
#         High price ($40): discharge → positive, charge → negative   ✓
#         Near p_ref:       hold → optimal (degradation makes small trades unprofitable)

#     Theoretical basis: potential-based reward shaping preserves optimal policy.

# NORMALISER NOTE:
#     normaliser_stats.npz from the 2022-2025 run is still valid.
#     p2_build_dataset.py does NOT need to be rerun unless STAGE1_START changes.
# """

# import os
# import torch

# # ══════════════════════════════════════════════════════════
# # PATHS
# # ══════════════════════════════════════════════════════════
# DATA_ROOT      = "./data/processed"
# CHECKPOINT_DIR = "./checkpoints/stage1"
# LOG_DIR        = "./logs"

# # ══════════════════════════════════════════════════════════
# # COLUMN NAMES
# # ══════════════════════════════════════════════════════════

# # 7-dim price vector (rt_mcpc cols are all null pre-RTC+B → excluded)
# PRICE_COLS = [
#     "rt_lmp",
#     "dam_spp",
#     "dam_as_regup",
#     "dam_as_regdn",
#     "dam_as_rrs",
#     "dam_as_ecrs",
#     "dam_as_nsrs",
# ]

# # 7 system condition columns (concatenated after TTFE, not fed into it)
# SYSTEM_COLS = [
#     "total_load_mw",
#     "load_forecast_mw",
#     "wind_actual_mw",
#     "wind_forecast_mw",
#     "solar_actual_mw",
#     "solar_forecast_mw",
#     "net_load_mw",
# ]

# TIMESTAMP_COL = "__index__"

# # ══════════════════════════════════════════════════════════
# # DATE SPLITS
# # ══════════════════════════════════════════════════════════
# # Start from 2022 to exclude 2021 Winter Storm Uri (kurtosis 786 → 631).
# # Kurtosis is still ~631 — Huber loss is still necessary.
# STAGE1_START = "2022-01-01"
# STAGE1_END   = "2025-12-04"
# VAL_START    = "2025-10-01"

# # ══════════════════════════════════════════════════════════
# # BATTERY PHYSICAL PARAMETERS
# # ══════════════════════════════════════════════════════════
# BATTERY_CAP_MWH  = 100.0      # total energy capacity (MWh)
# BATTERY_POWER_MW = 25.0       # max charge/discharge rate (MW)
# EFFICIENCY       = 0.92       # round-trip efficiency
# SOC_MIN          = 0.05       # minimum state-of-charge (fraction)
# SOC_MAX          = 0.95       # maximum state-of-charge (fraction)
# INTERVAL_H       = 5 / 60     # 5-minute intervals in hours

# # ══════════════════════════════════════════════════════════
# # MODEL ARCHITECTURE
# # ══════════════════════════════════════════════════════════
# # WINDOW_LEN=288 tested and failed (5x worse revenue, 4x worse critic).
# # Root cause unclear. Keep at 32 until reward/projection fixes are confirmed working.
# # Sequence: fix projection → confirm Plan C works → then test 288.
# WINDOW_LEN   = 32
# PRICE_DIM    = len(PRICE_COLS)    # = 7
# SYSTEM_DIM   = len(SYSTEM_COLS)   # = 7
# TIME_DIM     = 6
# SOC_DIM      = 1
# TTFE_DIM     = 64
# OBS_DIM      = TTFE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM  # = 78

# TTFE_NHEAD   = 4
# TTFE_NLAYERS = 2
# TTFE_DROPOUT = 0.1
# HIDDEN_DIM   = 256
# CLIP_SIGMA   = 5.0

# # ══════════════════════════════════════════════════════════
# # REWARD DESIGN — Plan C: Inventory-Adjusted Reward
# # ══════════════════════════════════════════════════════════
# #
# # action convention:
# #   +1 = charge   (buy from grid, SoC increases)
# #   -1 = discharge (sell to grid, SoC decreases)
# #    0 = hold
# #
# # grid_mwh:
# #   positive = energy sold to grid  (discharge)
# #   negative = energy bought from grid (charge)
# #
# # shaped_reward = grid_mwh * (rt_lmp - p_ref) / REWARD_SCALE
# #               - CYCLE_COST_PER_MWH * |grid_mwh| / REWARD_SCALE
# #
# # This fixes the credit-assignment problem that caused policy collapse:
# #   Prior reward (grid_mwh * rt_lmp) made Q(discharge) > Q(charge) at ALL prices.
# #   Inventory-adjusted reward corrects this relative to the p_ref baseline.
# #
# REWARD_SCALE       = 100.0    # fixed divisor for Q-value stability
# CYCLE_COST_PER_MWH = 1.0      # battery degradation cost per MWh cycled
#                                # $1/MWh conservative (real: $5-15/MWh)
#                                # Increase to $3-5 if agent over-cycles

# # ══════════════════════════════════════════════════════════
# # SAC HYPERPARAMETERS
# # ══════════════════════════════════════════════════════════

# # Fixed alpha — NO automatic tuning.
# # Rationale: auto-tuning (SAC v2) collapsed alpha to 0.03 by step 15k in
# # every previous run, removing the entropy bonus and causing policy collapse.
# # ALPHA_FIXED = 0.2 is the last value where training showed healthy behaviour.
# # Paper (Li et al. 2024 TempDRL) has no η_α in Table I → SAC v1 with fixed temp.
# ALPHA_FIXED = 0.2

# # --- Replay buffers ---
# DEMO_BUFFER_SIZE  = 100_000
# AGENT_BUFFER_SIZE = 1_000_000

# # --- Demonstrations ---
# # 50k steps ÷ 288 steps/episode ≈ 173 diverse episodes
# DEMO_STEPS       = 50_000
# DEMO_FLOOR       = 0.05       # never decays to 0 — preserves charge/discharge diversity
# DEMO_DECAY_STEPS = 200_000    # slow decay gives policy time to learn before demos fade

# # --- Critic stability ---
# # Huber loss: errors < HUBER_DELTA → quadratic, errors > HUBER_DELTA → linear
# # Reduces spike gradient dominance from 40,580× to ~15×
# HUBER_DELTA  = 10.0

# # --- Learning rates ---
# LR_ACTOR  = 3e-4
# LR_CRITIC = 1e-4    # lowered from 3e-4; critic climbed steadily before explosion
# LR_ALPHA  = 3e-4    # kept for reference only — NOT used (alpha is fixed)

# # --- Gradient clipping ---
# GRAD_CLIP = 0.5     # tightened from 1.0; prevents large updates during spike transitions

# # --- SAC core ---
# BATCH_SIZE     = 256
# GAMMA          = 0.99
# TAU            = 0.005
# TARGET_ENTROPY = -0.5    # kept for reference only — NOT used (alpha is fixed)

# # --- Episode and training length ---
# MAX_EP_STEPS = 288       # one trading day = 288 five-minute intervals
# TOTAL_STEPS  = 500_000   # run to 10k-20k first to confirm health, then extend

# # --- Logging and saving ---
# LOG_EVERY  = 1_000
# SAVE_EVERY = 50_000
# EVAL_EVERY = 10_000

# # ══════════════════════════════════════════════════════════
# # EARLY STOPPING CRITERIA
# # ══════════════════════════════════════════════════════════
# # Based on Stage 1 Plan A/B diagnostics:
# #   - Critic exploded at step 380k (loss ~50 → ~1000)
# #   - Policy collapsed (log_pi = +0.97, should be in [-2, 0])
# CRITIC_LOSS_STOP     = 300    # stop if 100-step MA of critic loss > 300
# LOG_PI_STOP          = 0.0    # stop if 100-step MA of log_pi > 0.0
#                                # (log_pi > 0 is theoretically impossible for healthy policy)
# MIN_STEP_BEFORE_STOP = 50_000  # don't trigger early stop before this step
# CHARGE_FRAC_MIN      = 0.05   # warn if charge fraction < 5% over last 1000 steps

# # ══════════════════════════════════════════════════════════
# # DEVICE
# # ══════════════════════════════════════════════════════════
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ══════════════════════════════════════════════════════════
# # SANITY CHECKS
# # ══════════════════════════════════════════════════════════
# assert len(PRICE_COLS)  == PRICE_DIM
# assert len(SYSTEM_COLS) == 7,  f"Need 7 system cols, got {len(SYSTEM_COLS)}"
# assert OBS_DIM          == 78, f"OBS_DIM should be 78, got {OBS_DIM}"
# assert WINDOW_LEN       == 32
# assert DEMO_FLOOR       >  0,  "DEMO_FLOOR must be > 0"
# assert HUBER_DELTA      >  0,  "HUBER_DELTA must be positive"
# assert ALPHA_FIXED      >  0,  "ALPHA_FIXED must be positive"
# assert CYCLE_COST_PER_MWH >= 0, "Degradation cost must be non-negative"

# if __name__ == "__main__":
#     print("Config loaded successfully.")
#     print(f"  STAGE1_START       : {STAGE1_START}  (excludes 2021 Winter Storm Uri)")
#     print(f"  PRICE_COLS  ({PRICE_DIM})    : {PRICE_COLS}")
#     print(f"  SYSTEM_COLS ({SYSTEM_DIM})   : {SYSTEM_COLS}")
#     print(f"  OBS_DIM            : {OBS_DIM}")
#     print(f"  WINDOW_LEN         : {WINDOW_LEN}")
#     print(f"  DEVICE             : {DEVICE}")
#     print(f"  ALPHA_FIXED        : {ALPHA_FIXED}  (no auto-tuning)")
#     print(f"  LR_CRITIC          : {LR_CRITIC}  (lowered from 3e-4)")
#     print(f"  GRAD_CLIP          : {GRAD_CLIP}  (tightened from 1.0)")
#     print(f"  HUBER_DELTA        : {HUBER_DELTA}")
#     print(f"  CYCLE_COST_PER_MWH : {CYCLE_COST_PER_MWH}  (degradation shaping)")
#     print(f"  REWARD_SCALE       : {REWARD_SCALE}")
#     print(f"  DEMO_STEPS         : {DEMO_STEPS:,}")
#     print(f"  DEMO_FLOOR         : {DEMO_FLOOR}  (never decays to 0)")
#     print()
#     print("  PROJECTION FIXES ACTIVE:")
#     print("    - feasible actions in critic target (not raw actor output)")
#     print("    - feasible actions in actor update  (not raw actor output)")
#     print("    - feasible actions stored in demo buffer")
#     print()
#     print("  Run order:")
#     print("    python pipeline/p2_build_dataset.py  (only if STAGE1_START changed)")
#     print("    python pipeline/p3_models.py")
#     print("    python pipeline/p6_reward_sanity.py  (MUST pass before training)")
#     print("    python pipeline/p4_train.py")

"""
config.py — Central Configuration for Stage 1
================================================
Plan C + Fixed Alpha + Projection Fixes + Proper Train/Val/Test Split.

THREE-WAY DATA SPLIT:
    Train : [STAGE1_START, VAL_START)    → gradient updates + normaliser
    Val   : [VAL_START, TEST_START)      → checkpoint selection during training
    Test  : [TEST_START, STAGE1_END]     → held out, final evaluation ONLY

    CRITICAL: Never evaluate on val split for final reported results.
    stage1_best.pt is selected based on val performance.
    p5_evaluate.py must use ERCOTDataset("test"), not ERCOTDataset("val").

REWARD FORMULA (Plan C — inventory-adjusted, Ng et al. 1999):
    grid_mwh = -action * BATTERY_POWER_MW * INTERVAL_H
    reward   = grid_mwh * (rt_lmp - p_ref) / REWARD_SCALE
             - CYCLE_COST_PER_MWH * |grid_mwh| / REWARD_SCALE

    p_ref = training median rt_lmp (computed from train split only)

    Low price ($9):   charge +$0.317, discharge -$0.317   ✓
    High price ($40): discharge +$0.329, charge -$0.329   ✓
    Near p_ref:       hold optimal (degradation cost)

NOTE: If VAL_START changed, rerun p2_build_dataset.py to recompute
      normaliser_stats.npz from the new training split.
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
# COLUMN NAMES
# ══════════════════════════════════════════════════════════

# 7-dim price vector (rt_mcpc cols are all null pre-RTC+B → excluded)
PRICE_COLS = [
    "rt_lmp",
    "dam_spp",
    "dam_as_regup",
    "dam_as_regdn",
    "dam_as_rrs",
    "dam_as_ecrs",
    "dam_as_nsrs",
]

# 7 system condition columns (concatenated after TTFE, not fed into it)
SYSTEM_COLS = [
    "total_load_mw",
    "load_forecast_mw",
    "wind_actual_mw",
    "wind_forecast_mw",
    "solar_actual_mw",
    "solar_forecast_mw",
    "net_load_mw",
]

TIMESTAMP_COL = "__index__"

# ══════════════════════════════════════════════════════════
# DATE SPLITS  — three-way split
# ══════════════════════════════════════════════════════════
# Excludes 2021 Winter Storm Uri (kurtosis 786 → 631 without 2021).
STAGE1_START = "2022-01-01"

# Val split: used for checkpoint selection ONLY during training.
# Never reported as final result.
VAL_START    = "2025-07-01"

# Test split: held out completely.
# Only touched in p5_evaluate.py for final reported numbers.
TEST_START   = "2025-10-01"

STAGE1_END   = "2025-12-04"

# ══════════════════════════════════════════════════════════
# BATTERY PHYSICAL PARAMETERS
# ══════════════════════════════════════════════════════════
BATTERY_CAP_MWH  = 100.0
BATTERY_POWER_MW = 25.0
EFFICIENCY       = 0.92
SOC_MIN          = 0.05
SOC_MAX          = 0.95
INTERVAL_H       = 5 / 60

# ══════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════
WINDOW_LEN   = 32
PRICE_DIM    = len(PRICE_COLS)    # = 7
SYSTEM_DIM   = len(SYSTEM_COLS)   # = 7
TIME_DIM     = 6
SOC_DIM      = 1
TTFE_DIM     = 64
OBS_DIM      = TTFE_DIM + SYSTEM_DIM + TIME_DIM + SOC_DIM  # = 78

TTFE_NHEAD   = 4
TTFE_NLAYERS = 2
TTFE_DROPOUT = 0.1
HIDDEN_DIM   = 256
CLIP_SIGMA   = 5.0

# ══════════════════════════════════════════════════════════
# REWARD DESIGN
# ══════════════════════════════════════════════════════════
REWARD_SCALE       = 100.0
CYCLE_COST_PER_MWH = 1.0

# ══════════════════════════════════════════════════════════
# SAC HYPERPARAMETERS
# ══════════════════════════════════════════════════════════

# Fixed alpha — no automatic tuning.
# Paper (Li et al. 2024 TempDRL) Table I has no η_α → SAC v1.
# Auto-tuning collapsed alpha to 0.03 in every previous run.
ALPHA_FIXED = 0.2

DEMO_BUFFER_SIZE  = 100_000
AGENT_BUFFER_SIZE = 1_000_000
DEMO_STEPS        = 50_000
DEMO_FLOOR        = 0.05
DEMO_DECAY_STEPS  = 200_000

HUBER_DELTA = 10.0
LR_ACTOR    = 3e-4
LR_CRITIC   = 1e-4
LR_ALPHA    = 3e-4    # reference only — NOT used (alpha is fixed)
GRAD_CLIP   = 0.5

BATCH_SIZE     = 256
GAMMA          = 0.99
TAU            = 0.005
TARGET_ENTROPY = -0.5  # reference only — NOT used (alpha is fixed)

MAX_EP_STEPS = 288
TOTAL_STEPS  = 500_000

LOG_EVERY  = 1_000
SAVE_EVERY = 50_000
EVAL_EVERY = 10_000

# ══════════════════════════════════════════════════════════
# EARLY STOPPING CRITERIA
# ══════════════════════════════════════════════════════════
# LOG_PI_STOP = 0.05 (not 0.0) — training stopped at step 130,954 because
# the 100-step MA of log_pi was +0.003, which is numerical noise.
# A threshold of 0.05 requires genuine policy instability before stopping.
CRITIC_LOSS_STOP         = 300
LOG_PI_STOP              = 0.05   # was 0.0 — raised to avoid noise-triggered stops
MIN_STEP_BEFORE_STOP     = 50_000
CHARGE_FRAC_MIN          = 0.05
EARLY_STOP_CONSEC_EVALS  = 2      # require N consecutive eval periods above threshold

# ══════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════
# SANITY CHECKS
# ══════════════════════════════════════════════════════════
assert len(PRICE_COLS)         == PRICE_DIM
assert len(SYSTEM_COLS)        == 7,   f"Need 7 system cols, got {len(SYSTEM_COLS)}"
assert OBS_DIM                 == 78,  f"OBS_DIM should be 78, got {OBS_DIM}"
assert WINDOW_LEN              == 32
assert DEMO_FLOOR              >  0,   "DEMO_FLOOR must be > 0"
assert HUBER_DELTA             >  0
assert ALPHA_FIXED             >  0
assert CYCLE_COST_PER_MWH      >= 0
assert LOG_PI_STOP             >= 0.0
assert EARLY_STOP_CONSEC_EVALS >= 1

if __name__ == "__main__":
    print("Config loaded.")
    print(f"  Train  : {STAGE1_START} → {VAL_START}  (gradient updates)")
    print(f"  Val    : {VAL_START} → {TEST_START}  (checkpoint selection)")
    print(f"  Test   : {TEST_START} → {STAGE1_END}  (held out — final eval only)")
    print(f"  ALPHA_FIXED        : {ALPHA_FIXED}")
    print(f"  LOG_PI_STOP        : {LOG_PI_STOP}  (raised from 0.0)")
    print(f"  EARLY_STOP_CONSEC  : {EARLY_STOP_CONSEC_EVALS} eval periods")
    print(f"  HUBER_DELTA        : {HUBER_DELTA}")
    print(f"  LR_CRITIC          : {LR_CRITIC}")
    print(f"  CYCLE_COST_PER_MWH : {CYCLE_COST_PER_MWH}")
    print()
    print("  IMPORTANT: If VAL_START changed, rerun p2_build_dataset.py")
    print("  to recompute normaliser_stats.npz for the new training split.")
