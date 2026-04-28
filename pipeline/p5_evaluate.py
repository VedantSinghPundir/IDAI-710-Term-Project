# # # # # """
# # # # # Pipeline 5 — Evaluation & Baselines
# # # # # ======================================
# # # # # Evaluates the trained Stage 1 agent against:
# # # # #   1. Rule-based heuristic  (charge below median, discharge above)
# # # # #   2. Perfect Information Optimisation (PIO) upper bound via LP
# # # # #   3. Trained SAC agent

# # # # # Prints a results table and saves results to logs/eval_results.csv.

# # # # # Usage:
# # # # #     python pipeline/p5_evaluate.py

# # # # # Requirements:
# # # # #     pip install scipy          (for LP baseline)
# # # # #     checkpoints/stage1/stage1_best.pt  must exist (run p4_train.py first)
# # # # # """

# # # # # import os
# # # # # import sys
# # # # # import numpy as np
# # # # # import pandas as pd

# # # # # sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# # # # # from pipeline.config import *
# # # # # from pipeline.p3_models import TTFE, Actor, Critic, FeasibilityProjection
# # # # # from pipeline.p4_train  import ERCOTDataset, ERCOTEnv, SACAgent, flatten_obs

# # # # # # Force deterministic episode starts for reproducible evaluation
# # # # # def _deterministic_reset(self):
# # # # #     self.idx      = WINDOW_LEN
# # # # #     self.soc      = 0.5
# # # # #     self.ep_steps = 0
# # # # #     return self._obs()
# # # # # ERCOTEnv.reset = _deterministic_reset

# # # # # # ════════════════════════════════════════════════════════
# # # # # # BASELINE 1 — Rule-Based Heuristic
# # # # # # ════════════════════════════════════════════════════════

# # # # # def run_rule_based(val_ds: ERCOTDataset) -> dict:
# # # # #     """
# # # # #     Simple price-threshold strategy:
# # # # #       - Charge at full rate when RT LMP < median price
# # # # #       - Discharge at full rate when RT LMP >= median price
# # # # #     No look-ahead; uses current price only.
# # # # #     """
# # # # #     # Compute median from training period
# # # # #     # train_lmp = val_ds.df[PRICE_COLS[0]]
# # # # #     # median_price = float(train_lmp.median())
# # # # #     train_ds = ERCOTDataset("train")
# # # # #     median_price = float(train_ds.df[PRICE_COLS[0]].median())

# # # # #     env   = ERCOTEnv(val_ds)
# # # # #     obs   = env.reset()
# # # # #     pw, sv, tf, soc_arr = obs
# # # # #     soc   = float(soc_arr[0])

# # # # #     total_rev = 0.0
# # # # #     revenues  = []
# # # # #     n_steps   = 0

# # # # #     while True:
# # # # #         # Get current RT LMP (raw, not normalised)
# # # # #         rt_lmp = val_ds.get_rt_lmp(env.idx)

# # # # #         if rt_lmp < median_price:
# # # # #             action = +1.0   # charge
# # # # #         else:
# # # # #             action = -1.0   # discharge

# # # # #         # Apply SoC limits manually
# # # # #         mwh_per_step = BATTERY_POWER_MW * INTERVAL_H
# # # # #         delta_soc = action * mwh_per_step / BATTERY_CAP_MWH
# # # # #         new_soc = np.clip(soc + delta_soc, SOC_MIN, SOC_MAX)
# # # # #         actual_action = (new_soc - soc) * BATTERY_CAP_MWH / mwh_per_step

# # # # #         (pw, sv, tf, soc_arr), reward, done = env.step(actual_action, new_soc)
# # # # #         soc = float(soc_arr[0])
# # # # #         # total_rev += reward
# # # # #         total_rev += reward * REWARD_SCALE
# # # # #         # revenues.append(reward)
# # # # #         revenues.append(reward * REWARD_SCALE)
# # # # #         n_steps += 1

# # # # #         if done:
# # # # #             break

# # # # #     return {
# # # # #         "name":         "Rule-based heuristic",
# # # # #         "total_rev":    total_rev,
# # # # #         "mean_rev":     np.mean(revenues),
# # # # #         "std_rev":      np.std(revenues),
# # # # #         "n_steps":      n_steps,
# # # # #     }


# # # # # # ════════════════════════════════════════════════════════
# # # # # # BASELINE 2 — Perfect Information Optimisation (PIO)
# # # # # # ════════════════════════════════════════════════════════

# # # # # def run_pio(val_ds: ERCOTDataset, max_steps: int = 5000) -> dict:
# # # # #     """
# # # # #     Linear Program with perfect price foresight.
# # # # #     This is the theoretical upper bound — the agent can never beat it.
# # # # #     Uses scipy.optimize.linprog.

# # # # #     Variables: charge_t, discharge_t ≥ 0 for each interval
# # # # #     Objective: maximise Σ price_t × (discharge_t - charge_t) × INTERVAL_H
# # # # #     """
# # # # #     try:
# # # # #         from scipy.optimize import linprog
# # # # #     except ImportError:
# # # # #         print("  [PIO] scipy not installed. Run: pip install scipy")
# # # # #         return {"name": "PIO (skip)", "total_rev": float("nan"),
# # # # #                 "mean_rev": float("nan"), "std_rev": float("nan"), "n_steps": 0}

# # # # #     T = min(max_steps, len(val_ds) - WINDOW_LEN - 1)
# # # # #     prices = val_ds.df[PRICE_COLS[0]].iloc[WINDOW_LEN:WINDOW_LEN + T].values

# # # # #     # Variables: [charge_0..T-1, discharge_0..T-1]
# # # # #     # Maximise revenue = price × (discharge - charge) × INTERVAL_H
# # # # #     # → minimise -price × (discharge - charge) × INTERVAL_H
# # # # #     rev_per_mw = prices * INTERVAL_H
# # # # #     c = np.concatenate([-rev_per_mw * (-1),   # charging costs (negative revenue)
# # # # #                          -rev_per_mw * ( 1)])  # discharging earns

# # # # #     # Bounds: 0 ≤ charge ≤ BATTERY_POWER_MW, 0 ≤ discharge ≤ BATTERY_POWER_MW
# # # # #     bounds = [(0, BATTERY_POWER_MW)] * T + [(0, BATTERY_POWER_MW)] * T

# # # # #     # SoC evolution: soc_t = soc_0 + Σ(charge - discharge) × INTERVAL_H / CAP
# # # # #     # SoC constraints: SOC_MIN ≤ soc_t ≤ SOC_MAX for all t
# # # # #     A_ub, b_ub_upper, b_ub_lower = [], [], []
# # # # #     coeff = INTERVAL_H / BATTERY_CAP_MWH
# # # # #     for t in range(T):
# # # # #         row = np.zeros(2 * T)
# # # # #         row[:t+1]   =  coeff    # charge cumsum
# # # # #         row[T:T+t+1]= -coeff    # discharge cumsum
# # # # #         A_ub.append(row)        # soc ≤ SOC_MAX
# # # # #         A_ub.append(-row)       # -soc ≤ -SOC_MIN

# # # # #     b_ub = np.array([SOC_MAX - 0.5] * T + [-(SOC_MIN - 0.5)] * T)

# # # # #     result = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub,
# # # # #                      bounds=bounds, method="highs")

# # # # #     if result.success:
# # # # #         charge    = result.x[:T]
# # # # #         discharge = result.x[T:]
# # # # #         revenues  = prices * (discharge - charge) * INTERVAL_H
# # # # #         total_rev = revenues.sum()
# # # # #     else:
# # # # #         print(f"  [PIO] LP did not converge: {result.message}")
# # # # #         total_rev = float("nan")
# # # # #         revenues  = np.zeros(T)

# # # # #     return {
# # # # #         "name":      "PIO (perfect foresight LP)",
# # # # #         "total_rev": float(total_rev),
# # # # #         "mean_rev":  float(np.mean(revenues)),
# # # # #         "std_rev":   float(np.std(revenues)),
# # # # #         "n_steps":   T,
# # # # #     }


# # # # # # ════════════════════════════════════════════════════════
# # # # # # TRAINED SAC AGENT EVALUATION
# # # # # # ════════════════════════════════════════════════════════

# # # # # def run_sac_agent(val_ds: ERCOTDataset, max_steps: int = 5000) -> dict:
# # # # #     best_ckpt = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")
# # # # #     if not os.path.exists(best_ckpt):
# # # # #         # Fallback to most recent checkpoint
# # # # #         ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
# # # # #         if not ckpts:
# # # # #             print("  [SAC] No checkpoint found. Run p4_train.py first.")
# # # # #             return {"name": "SAC agent (no ckpt)", "total_rev": float("nan"),
# # # # #                     "mean_rev": float("nan"), "std_rev": float("nan"), "n_steps": 0}
# # # # #         best_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])

# # # # #     agent = SACAgent()
# # # # #     step  = agent.load(best_ckpt)
# # # # #     print(f"  [SAC] Loaded checkpoint from step {step}")

# # # # #     env   = ERCOTEnv(val_ds)
# # # # #     obs   = env.reset()
# # # # #     pw, sv, tf, soc_arr = obs
# # # # #     soc_val = float(soc_arr[0])

# # # # #     revenues = []
# # # # #     n_steps  = 0

# # # # #     while n_steps < max_steps:
# # # # #         action, new_soc = agent.select_action(pw, sv, tf, soc_val, deterministic=True)
# # # # #         (pw, sv, tf, soc_arr), reward, done = env.step(action, new_soc)
# # # # #         soc_val = float(soc_arr[0])
# # # # #         # revenues.append(reward)
# # # # #         revenues.append(reward * REWARD_SCALE)
# # # # #         n_steps += 1
# # # # #         if done:
# # # # #             break

# # # # #     total_rev = float(np.sum(revenues))
# # # # #     return {
# # # # #         "name":      f"SAC agent (step {step})",
# # # # #         "total_rev": total_rev,
# # # # #         "mean_rev":  float(np.mean(revenues)),
# # # # #         "std_rev":   float(np.std(revenues)),
# # # # #         "n_steps":   n_steps,
# # # # #     }


# # # # # # ════════════════════════════════════════════════════════
# # # # # # PRINT RESULTS TABLE
# # # # # # ════════════════════════════════════════════════════════

# # # # # def print_results(results: list[dict], pio_rev: float):
# # # # #     print("\n" + "=" * 65)
# # # # #     print("Evaluation Results")
# # # # #     print("=" * 65)
# # # # #     print(f"  {'Method':<35} {'Total Rev ($)':>14} {'vs Heuristic':>13} {'% of PIO':>10}")
# # # # #     print(f"  {'-'*65}")

# # # # #     heuristic_rev = next((r["total_rev"] for r in results
# # # # #                           if "heuristic" in r["name"].lower()), 0)

# # # # #     for r in results:
# # # # #         rev = r["total_rev"]
# # # # #         if np.isnan(rev):
# # # # #             rev_str = "n/a"
# # # # #             vs_str  = "n/a"
# # # # #             pio_str = "n/a"
# # # # #         else:
# # # # #             rev_str = f"${rev:>12,.2f}"
# # # # #             vs_str  = f"{(rev/heuristic_rev - 1)*100:>+.1f}%" if heuristic_rev else "n/a"
# # # # #             pio_str = f"{rev/pio_rev*100:>8.1f}%"  if not np.isnan(pio_rev) and pio_rev else "n/a"
# # # # #         print(f"  {r['name']:<35} {rev_str:>14} {vs_str:>13} {pio_str:>10}")

# # # # #     print(f"\n  Note: PIO is the theoretical max (perfect foresight).")
# # # # #     print(f"  SAC should exceed rule-based and approach PIO over training.")


# # # # # # ════════════════════════════════════════════════════════
# # # # # # MAIN
# # # # # # ════════════════════════════════════════════════════════

# # # # # def main():
# # # # #     print("=" * 60)
# # # # #     print("Pipeline 5 — Evaluation")
# # # # #     print("=" * 60)

# # # # #     val_ds  = ERCOTDataset("val")
# # # # #     # MAX_STEPS = 5000
# # # # #     # MAX_STEPS = 18000   # full val period
# # # # #     MAX_STEPS_EVAL = 18000   # heuristic + SAC on full val period
# # # # #     MAX_STEPS_PIO  = 2000    # PIO is reference only — keep small


# # # # #     print("\n[1] Running rule-based heuristic...")
# # # # #     rule = run_rule_based(val_ds)
# # # # #     print(f"    Total revenue: ${rule['total_rev']:,.2f}")

# # # # #     print("\n[2] Running Perfect Information LP (PIO upper bound)...")
# # # # #     # pio  = run_pio(val_ds, max_steps=MAX_STEPS)
# # # # #     pio  = run_pio(val_ds, max_steps=MAX_STEPS_PIO)
# # # # #     print(f"    Total revenue: ${pio['total_rev']:,.2f}")

# # # # #     print("\n[3] Running trained SAC agent (deterministic)...")
# # # # #     # sac  = run_sac_agent(val_ds, max_steps=MAX_STEPS)
# # # # #     sac  = run_sac_agent(val_ds, max_steps=MAX_STEPS_EVAL)
# # # # #     print(f"    Total revenue: ${sac['total_rev']:,.2f}")

# # # # #     results = [rule, pio, sac]
# # # # #     print_results(results, pio["total_rev"])

# # # # #     # Save to CSV
# # # # #     os.makedirs(LOG_DIR, exist_ok=True)
# # # # #     out = pd.DataFrame(results)
# # # # #     out.to_csv(os.path.join(LOG_DIR, "eval_results.csv"), index=False)
# # # # #     print(f"\n  Results saved → {LOG_DIR}/eval_results.csv")

# # # # #     print("\n" + "=" * 60)
# # # # #     print("✓ Evaluation complete.")
# # # # #     print("Share these numbers with your teammates as your Stage 1 result.")


# # # # # if __name__ == "__main__":
# # # # #     main()


# # # # """
# # # # Pipeline 5 — Evaluation & Baselines (Inventory-Adjusted)
# # # # ==========================================================
# # # # Evaluates trained agent against rule-based heuristic and PIO.

# # # # KEY CHANGE: Reports BOTH cash revenue AND inventory-adjusted profit.
# # # # This matters because:
# # # #   - Cash only: unfair if agent ends with different SoC than it started
# # # #   - If agent charged a lot but episode ended before selling → cash looks bad
# # # #   - If agent sold pre-existing stored energy → cash looks artificially good
# # # #   - Inventory-adjusted profit = cash + value of SoC change - degradation

# # # # EVALUATION REWARD:
# # # #   Uses REAL DOLLARS (no inventory shaping) for cash revenue column.
# # # #   Uses inventory-adjusted profit for fair comparison column.
# # # #   The agent was TRAINED on shaped rewards but is EVALUATED on real dollars.

# # # # Usage:
# # # #     python pipeline/p5_evaluate.py

# # # # Requirements:
# # # #     pip install scipy
# # # #     checkpoints/stage1/stage1_best.pt  (run p4_train.py first)
# # # # """

# # # # import os
# # # # import sys
# # # # import numpy as np
# # # # import pandas as pd

# # # # sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# # # # from pipeline.config import *
# # # # from pipeline.p3_models import TTFE, Actor, Critic, FeasibilityProjection
# # # # from pipeline.p4_train  import (ERCOTDataset, ERCOTEnv, SACAgent,
# # # #                                  flatten_obs)

# # # # # Force deterministic evaluation starts
# # # # def _deterministic_reset(self):
# # # #     self.idx      = WINDOW_LEN
# # # #     self.soc      = 0.5
# # # #     self.ep_steps = 0
# # # #     return self._obs()

# # # # # Force evaluation step — no episode limit (runs full evaluation window)
# # # # def _eval_step(self, action: float, new_soc: float):
# # # #     rt_lmp     = self.ds.get_rt_lmp(self.idx)
# # # #     grid_mwh   = -action * BATTERY_POWER_MW * INTERVAL_H
# # # #     cash_reward = grid_mwh * rt_lmp     # real dollars — no shaping
# # # #     degradation = CYCLE_COST_PER_MWH * abs(grid_mwh)

# # # #     self.soc       = new_soc
# # # #     self.idx      += 1
# # # #     self.ep_steps += 1
# # # #     done = (self.idx >= len(self.ds) - 1)  # only end at dataset boundary
# # # #     return self._obs(), cash_reward, done, degradation

# # # # ERCOTEnv.reset = _deterministic_reset
# # # # ERCOTEnv.step  = _eval_step


# # # # # ════════════════════════════════════════════════════════
# # # # # HELPER — Inventory-adjusted profit
# # # # # ════════════════════════════════════════════════════════

# # # # def inventory_adjusted_profit(cash_revenue: float,
# # # #                                initial_soc: float,
# # # #                                final_soc: float,
# # # #                                total_degradation: float,
# # # #                                p_ref: float) -> float:
# # # #     """
# # # #     Accounts for energy bought but not yet sold (or sold from initial store).

# # # #     inventory_change = (final_soc - initial_soc) × capacity × p_ref
# # # #     If agent ends more charged than it started → positive (stored energy has value)
# # # #     If agent ends more discharged than it started → negative (sold from store)

# # # #     Fair comparison requires all agents start and end at same SoC,
# # # #     OR we use this inventory-adjusted metric.
# # # #     """
# # # #     inventory_change = (final_soc - initial_soc) * BATTERY_CAP_MWH * p_ref
# # # #     return cash_revenue + inventory_change - total_degradation


# # # # # ════════════════════════════════════════════════════════
# # # # # BASELINE 1 — Rule-Based Heuristic
# # # # # ════════════════════════════════════════════════════════

# # # # def run_rule_based(val_ds: ERCOTDataset, max_steps: int = 18000) -> dict:
# # # #     """
# # # #     Charge when rt_lmp < training median, discharge when >= median.
# # # #     Uses training median as threshold (same as agent's p_ref — fair comparison).
# # # #     """
# # # #     train_ds     = ERCOTDataset("train")
# # # #     median_price = float(train_ds.df[PRICE_COLS[0]].median())
# # # #     p_ref        = median_price   # same reference as training

# # # #     env = ERCOTEnv(val_ds)
# # # #     obs = env.reset()
# # # #     pw, sv, tf, soc_arr = obs
# # # #     soc = float(soc_arr[0])
# # # #     initial_soc = soc

# # # #     total_cash        = 0.0
# # # #     total_degradation = 0.0
# # # #     n_steps           = 0
# # # #     proj              = FeasibilityProjection()

# # # #     while n_steps < max_steps:
# # # #         rt_lmp = val_ds.get_rt_lmp(env.idx)
# # # #         action = -1.0 if rt_lmp >= median_price else 1.0

# # # #         mwh_per_step = BATTERY_POWER_MW * INTERVAL_H
# # # #         delta_soc    = action * mwh_per_step / BATTERY_CAP_MWH
# # # #         new_soc      = np.clip(soc + delta_soc, SOC_MIN, SOC_MAX)
# # # #         actual_action = (new_soc - soc) * BATTERY_CAP_MWH / mwh_per_step

# # # #         (pw, sv, tf, soc_arr), cash_reward, done, degradation = env.step(actual_action, new_soc)
# # # #         soc = float(soc_arr[0])

# # # #         total_cash        += cash_reward
# # # #         total_degradation += degradation
# # # #         n_steps           += 1

# # # #         if done:
# # # #             break

# # # #     final_soc  = soc
# # # #     inv_profit = inventory_adjusted_profit(
# # # #         total_cash, initial_soc, final_soc, total_degradation, p_ref
# # # #     )

# # # #     return {
# # # #         "name":              "Rule-based heuristic",
# # # #         "cash_revenue":      total_cash,
# # # #         "inv_adjusted":      inv_profit,
# # # #         "inventory_change":  (final_soc - initial_soc) * BATTERY_CAP_MWH * p_ref,
# # # #         "degradation_cost":  total_degradation,
# # # #         "initial_soc":       initial_soc,
# # # #         "final_soc":         final_soc,
# # # #         "n_steps":           n_steps,
# # # #         "p_ref":             p_ref,
# # # #     }


# # # # # ════════════════════════════════════════════════════════
# # # # # BASELINE 2 — PIO (Perfect Information Optimisation)
# # # # # ════════════════════════════════════════════════════════

# # # # def run_pio(val_ds: ERCOTDataset, max_steps: int = 2000) -> dict:
# # # #     """
# # # #     LP with perfect price foresight — theoretical upper bound.
# # # #     Reports cash revenue only (PIO does not have degradation model).
# # # #     Runs max_steps only — serves as reference ceiling.
# # # #     """
# # # #     try:
# # # #         from scipy.optimize import linprog
# # # #     except ImportError:
# # # #         print("  [PIO] scipy not installed.")
# # # #         return {"name": "PIO (skip)", "cash_revenue": float("nan"),
# # # #                 "inv_adjusted": float("nan"), "n_steps": 0,
# # # #                 "initial_soc": 0.5, "final_soc": 0.5,
# # # #                 "inventory_change": 0.0, "degradation_cost": 0.0,
# # # #                 "p_ref": 0.0}

# # # #     T      = min(max_steps, len(val_ds) - WINDOW_LEN - 1)
# # # #     prices = val_ds.df[PRICE_COLS[0]].iloc[WINDOW_LEN:WINDOW_LEN + T].values

# # # #     rev_per_mw = prices * INTERVAL_H
# # # #     c = np.concatenate([-rev_per_mw * (-1), -rev_per_mw * (1)])
# # # #     bounds = [(0, BATTERY_POWER_MW)] * T + [(0, BATTERY_POWER_MW)] * T

# # # #     coeff = INTERVAL_H / BATTERY_CAP_MWH
# # # #     A_ub  = []
# # # #     for t in range(T):
# # # #         row = np.zeros(2 * T)
# # # #         row[:t+1]    =  coeff
# # # #         row[T:T+t+1] = -coeff
# # # #         A_ub.append(row)
# # # #         A_ub.append(-row)
# # # #     b_ub = np.array([SOC_MAX - 0.5] * T + [-(SOC_MIN - 0.5)] * T)

# # # #     result = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub,
# # # #                      bounds=bounds, method="highs")

# # # #     if result.success:
# # # #         charge    = result.x[:T]
# # # #         discharge = result.x[T:]
# # # #         revenues  = prices * (discharge - charge) * INTERVAL_H
# # # #         total_rev = float(revenues.sum())
# # # #         final_soc_change = float((charge.sum() - discharge.sum()) * INTERVAL_H / BATTERY_CAP_MWH)
# # # #     else:
# # # #         print(f"  [PIO] LP failed: {result.message}")
# # # #         total_rev = float("nan")
# # # #         final_soc_change = 0.0

# # # #     return {
# # # #         "name":             "PIO (perfect foresight LP)",
# # # #         "cash_revenue":     total_rev,
# # # #         "inv_adjusted":     total_rev,   # PIO optimises cash directly
# # # #         "inventory_change": final_soc_change * BATTERY_CAP_MWH * 24.21,
# # # #         "degradation_cost": 0.0,
# # # #         "initial_soc":      0.5,
# # # #         "final_soc":        0.5 + final_soc_change,
# # # #         "n_steps":          T,
# # # #         "p_ref":            24.21,
# # # #     }


# # # # # ════════════════════════════════════════════════════════
# # # # # TRAINED SAC AGENT
# # # # # ════════════════════════════════════════════════════════

# # # # def run_sac_agent(val_ds: ERCOTDataset, max_steps: int = 18000) -> dict:
# # # #     """
# # # #     Deterministic rollout of trained SAC agent.
# # # #     Reports cash revenue AND inventory-adjusted profit.
# # # #     """
# # # #     best_ckpt = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")
# # # #     if not os.path.exists(best_ckpt):
# # # #         ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
# # # #         if not ckpts:
# # # #             print("  [SAC] No checkpoint found.")
# # # #             return {"name": "SAC (no ckpt)", "cash_revenue": float("nan"),
# # # #                     "inv_adjusted": float("nan"), "n_steps": 0,
# # # #                     "initial_soc": 0.5, "final_soc": 0.5,
# # # #                     "inventory_change": 0.0, "degradation_cost": 0.0, "p_ref": 0.0}
# # # #         best_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])

# # # #     agent = SACAgent()
# # # #     step  = agent.load(best_ckpt)
# # # #     print(f"  [SAC] Loaded checkpoint from step {step}")

# # # #     # Get p_ref from training data
# # # #     train_ds = ERCOTDataset("train")
# # # #     p_ref    = float(train_ds.df[PRICE_COLS[0]].median())

# # # #     env = ERCOTEnv(val_ds)
# # # #     obs = env.reset()
# # # #     pw, sv, tf, soc_arr = obs
# # # #     soc_val     = float(soc_arr[0])
# # # #     initial_soc = soc_val

# # # #     total_cash        = 0.0
# # # #     total_degradation = 0.0
# # # #     n_steps           = 0

# # # #     while n_steps < max_steps:
# # # #         action, new_soc = agent.select_action(pw, sv, tf, soc_val, deterministic=True)
# # # #         (pw, sv, tf, soc_arr), cash_reward, done, degradation = env.step(action, new_soc)
# # # #         soc_val = float(soc_arr[0])

# # # #         total_cash        += cash_reward
# # # #         total_degradation += degradation
# # # #         n_steps           += 1

# # # #         if done:
# # # #             break

# # # #     final_soc  = soc_val
# # # #     inv_profit = inventory_adjusted_profit(
# # # #         total_cash, initial_soc, final_soc, total_degradation, p_ref
# # # #     )

# # # #     return {
# # # #         "name":             f"SAC agent (step {step})",
# # # #         "cash_revenue":     total_cash,
# # # #         "inv_adjusted":     inv_profit,
# # # #         "inventory_change": (final_soc - initial_soc) * BATTERY_CAP_MWH * p_ref,
# # # #         "degradation_cost": total_degradation,
# # # #         "initial_soc":      initial_soc,
# # # #         "final_soc":        final_soc,
# # # #         "n_steps":          n_steps,
# # # #         "p_ref":            p_ref,
# # # #     }


# # # # # ════════════════════════════════════════════════════════
# # # # # PRINT RESULTS
# # # # # ════════════════════════════════════════════════════════

# # # # def print_results(results: list):
# # # #     pio_rev = next((r["inv_adjusted"] for r in results
# # # #                     if "PIO" in r["name"]), float("nan"))
# # # #     heuristic_inv = next((r["inv_adjusted"] for r in results
# # # #                           if "heuristic" in r["name"].lower()), 0.0)

# # # #     print("\n" + "=" * 80)
# # # #     print("Evaluation Results")
# # # #     print("=" * 80)
# # # #     print(f"  Note: p_ref = training median price (used for inventory adjustment)")
# # # #     print(f"  All methods evaluated from same start: idx=WINDOW_LEN, SoC=0.5")
# # # #     print()
# # # #     print(f"  {'Method':<35} {'Cash ($)':>12} {'Inv-Adj ($)':>12} "
# # # #           f"{'vs Heuristic':>13} {'Final SoC':>10}")
# # # #     print(f"  {'-'*82}")

# # # #     for r in results:
# # # #         cash = r["cash_revenue"]
# # # #         inv  = r["inv_adjusted"]
# # # #         fsoc = r["final_soc"]
# # # #         n    = r["n_steps"]

# # # #         if np.isnan(inv):
# # # #             print(f"  {r['name']:<35} {'n/a':>12} {'n/a':>12} {'n/a':>13} {'n/a':>10}")
# # # #         else:
# # # #             vs_h = f"{(inv/heuristic_inv - 1)*100:>+.1f}%" if heuristic_inv else "n/a"
# # # #             print(f"  {r['name']:<35} ${cash:>10,.2f} ${inv:>10,.2f} "
# # # #                   f"{vs_h:>13} {fsoc:>10.3f}  ({n} steps)")

# # # #     print()
# # # #     print(f"  {'Key insight:'}")
# # # #     print(f"  Cash revenue = raw market earnings")
# # # #     print(f"  Inv-adjusted = cash + SoC change × p_ref - degradation")
# # # #     print(f"  Use inv-adjusted for fair comparison (accounts for stored energy)")
# # # #     print()
# # # #     heur = next((r for r in results if "heuristic" in r["name"].lower()), None)
# # # #     sac  = next((r for r in results if "SAC" in r["name"]), None)
# # # #     if heur and sac:
# # # #         print(f"  Heuristic final SoC: {heur['final_soc']:.3f}  "
# # # #               f"(started at {heur['initial_soc']:.3f})")
# # # #         print(f"  SAC agent final SoC: {sac['final_soc']:.3f}  "
# # # #               f"(started at {sac['initial_soc']:.3f})")
# # # #         if abs(heur['final_soc'] - sac['final_soc']) > 0.1:
# # # #             print(f"  [NOTE] Final SoC differs by >{abs(heur['final_soc']-sac['final_soc']):.2f} "
# # # #                   f"— inventory adjustment matters for fair comparison")


# # # # # ════════════════════════════════════════════════════════
# # # # # MAIN
# # # # # ════════════════════════════════════════════════════════

# # # # def main():
# # # #     print("=" * 60)
# # # #     print("Pipeline 5 — Evaluation (Inventory-Adjusted)")
# # # #     print("=" * 60)

# # # #     val_ds = ERCOTDataset("val")

# # # #     MAX_STEPS_EVAL = 18000   # heuristic + SAC: full val period
# # # #     MAX_STEPS_PIO  = 2000    # PIO: reference only (LP is slow)

# # # #     print("\n[1] Running rule-based heuristic...")
# # # #     rule = run_rule_based(val_ds, max_steps=MAX_STEPS_EVAL)
# # # #     print(f"    Cash: ${rule['cash_revenue']:,.2f} | "
# # # #           f"Inv-adj: ${rule['inv_adjusted']:,.2f} | "
# # # #           f"Final SoC: {rule['final_soc']:.3f}")

# # # #     print("\n[2] Running PIO upper bound (2000 steps, reference only)...")
# # # #     pio  = run_pio(val_ds, max_steps=MAX_STEPS_PIO)
# # # #     print(f"    Cash: ${pio['cash_revenue']:,.2f}")

# # # #     print("\n[3] Running trained SAC agent (deterministic)...")
# # # #     sac  = run_sac_agent(val_ds, max_steps=MAX_STEPS_EVAL)
# # # #     print(f"    Cash: ${sac['cash_revenue']:,.2f} | "
# # # #           f"Inv-adj: ${sac['inv_adjusted']:,.2f} | "
# # # #           f"Final SoC: {sac['final_soc']:.3f}")

# # # #     results = [rule, pio, sac]
# # # #     print_results(results)

# # # #     # Save to CSV
# # # #     os.makedirs(LOG_DIR, exist_ok=True)
# # # #     out = pd.DataFrame([{
# # # #         "name":         r["name"],
# # # #         "cash_revenue": r["cash_revenue"],
# # # #         "inv_adjusted": r["inv_adjusted"],
# # # #         "final_soc":    r["final_soc"],
# # # #         "n_steps":      r["n_steps"],
# # # #     } for r in results])
# # # #     out.to_csv(os.path.join(LOG_DIR, "eval_results.csv"), index=False)
# # # #     print(f"\n  Results saved → {LOG_DIR}/eval_results.csv")

# # # #     print("\n" + "=" * 60)
# # # #     print("✓ Evaluation complete.")


# # # # if __name__ == "__main__":
# # # #     main()

# # # """
# # # Pipeline 5 — Evaluation & Baselines (Plan D)
# # # =============================================
# # # Evaluates trained agent against rule-based heuristic and PIO.

# # # EVALUATION REWARD:
# # #   Uses REAL CASH ONLY (no shaping) for fair comparison.
# # #   Heuristic uses EMA-based threshold (same as training — fair).
# # #   SAC agent evaluated deterministically from fixed start.

# # # Reports:
# # #   - Cash revenue: raw market earnings in real dollars
# # #   - Charge fraction: how often the agent charged vs discharged
# # #   - Final SoC: where the battery ended up

# # # Usage:
# # #     python pipeline/p5_evaluate.py
# # # """

# # # import os
# # # import sys
# # # import numpy as np
# # # import pandas as pd

# # # sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# # # from pipeline.config import *
# # # from pipeline.p3_models import TTFE, Actor, Critic, FeasibilityProjection
# # # from pipeline.p4_train  import ERCOTDataset, ERCOTEnv, SACAgent, flatten_obs


# # # # Patch for evaluation: deterministic start, no episode limit, cash reward only
# # # def _eval_reset(self):
# # #     self.idx      = WINDOW_LEN
# # #     self.soc      = 0.5
# # #     self.ep_steps = 0
# # #     self.ema      = None
# # #     return self._obs()


# # # def _eval_step(self, action: float, new_soc: float):
# # #     """Evaluation step — cash only, no episode limit, EMA still tracked."""
# # #     rt_lmp = self.ds.get_rt_lmp(self.idx)

# # #     # Still update EMA for fair comparison with training
# # #     if self.ema is None:
# # #         self.ema = rt_lmp
# # #     else:
# # #         self.ema = TAU_S * self.ema + (1.0 - TAU_S) * rt_lmp

# # #     grid_mwh    = -action * BATTERY_POWER_MW * INTERVAL_H
# # #     cash_reward = grid_mwh * rt_lmp   # real dollars, no shaping

# # #     self.soc       = new_soc
# # #     self.idx      += 1
# # #     self.ep_steps += 1
# # #     done = (self.idx >= len(self.ds) - 1)   # only dataset boundary
# # #     return self._obs(), cash_reward, done


# # # ERCOTEnv.reset = _eval_reset
# # # ERCOTEnv.step  = _eval_step


# # # # ════════════════════════════════════════════════════════
# # # # BASELINE 1 — EMA-Based Heuristic
# # # # ════════════════════════════════════════════════════════

# # # def run_rule_based(val_ds: ERCOTDataset, max_steps: int = 18000) -> dict:
# # #     """
# # #     Charge when rt_lmp < EMA, discharge when rt_lmp >= EMA.
# # #     Uses same TAU_S=0.9 EMA as training — fair comparison.
# # #     No look-ahead bias — EMA computed from real-time stream only.
# # #     """
# # #     env = ERCOTEnv(val_ds)
# # #     obs = env.reset()
# # #     pw, sv, tf, soc_arr = obs
# # #     soc = float(soc_arr[0])
# # #     initial_soc = soc

# # #     total_cash    = 0.0
# # #     charge_count  = 0
# # #     n_steps       = 0

# # #     while n_steps < max_steps:
# # #         rt_lmp = val_ds.get_rt_lmp(env.idx)

# # #         # Use EMA threshold (same as training)
# # #         threshold = env.ema if env.ema is not None else rt_lmp
# # #         action    = -1.0 if rt_lmp >= threshold else 1.0

# # #         mwh_per_step  = BATTERY_POWER_MW * INTERVAL_H
# # #         delta_soc     = action * mwh_per_step / BATTERY_CAP_MWH
# # #         new_soc       = np.clip(soc + delta_soc, SOC_MIN, SOC_MAX)
# # #         actual_action = (new_soc - soc) * BATTERY_CAP_MWH / mwh_per_step

# # #         (pw, sv, tf, soc_arr), cash_reward, done = env.step(actual_action, new_soc)
# # #         soc = float(soc_arr[0])

# # #         total_cash += cash_reward
# # #         if actual_action > 0:
# # #             charge_count += 1
# # #         n_steps += 1

# # #         if done:
# # #             break

# # #     return {
# # #         "name":          "EMA heuristic (TAU_S=0.9)",
# # #         "cash_revenue":  total_cash,
# # #         "charge_frac":   charge_count / max(n_steps, 1),
# # #         "initial_soc":   initial_soc,
# # #         "final_soc":     soc,
# # #         "n_steps":       n_steps,
# # #     }


# # # # ════════════════════════════════════════════════════════
# # # # BASELINE 1b — Global Median Heuristic (for comparison)
# # # # ════════════════════════════════════════════════════════

# # # def run_global_median_heuristic(val_ds: ERCOTDataset,
# # #                                 max_steps: int = 18000) -> dict:
# # #     """
# # #     Original heuristic using global training median.
# # #     Kept for comparison — shows the look-ahead bias advantage.
# # #     """
# # #     train_ds     = ERCOTDataset("train")
# # #     median_price = float(train_ds.df[PRICE_COLS[0]].median())

# # #     env = ERCOTEnv(val_ds)
# # #     obs = env.reset()
# # #     pw, sv, tf, soc_arr = obs
# # #     soc = float(soc_arr[0])
# # #     initial_soc = soc

# # #     total_cash   = 0.0
# # #     charge_count = 0
# # #     n_steps      = 0

# # #     while n_steps < max_steps:
# # #         rt_lmp = val_ds.get_rt_lmp(env.idx)
# # #         action = -1.0 if rt_lmp >= median_price else 1.0

# # #         mwh_per_step  = BATTERY_POWER_MW * INTERVAL_H
# # #         delta_soc     = action * mwh_per_step / BATTERY_CAP_MWH
# # #         new_soc       = np.clip(soc + delta_soc, SOC_MIN, SOC_MAX)
# # #         actual_action = (new_soc - soc) * BATTERY_CAP_MWH / mwh_per_step

# # #         (pw, sv, tf, soc_arr), cash_reward, done = env.step(actual_action, new_soc)
# # #         soc = float(soc_arr[0])

# # #         total_cash += cash_reward
# # #         if actual_action > 0:
# # #             charge_count += 1
# # #         n_steps += 1

# # #         if done:
# # #             break

# # #     return {
# # #         "name":         f"Global median heuristic (${median_price:.2f})",
# # #         "cash_revenue": total_cash,
# # #         "charge_frac":  charge_count / max(n_steps, 1),
# # #         "initial_soc":  initial_soc,
# # #         "final_soc":    soc,
# # #         "n_steps":      n_steps,
# # #     }


# # # # ════════════════════════════════════════════════════════
# # # # BASELINE 2 — PIO
# # # # ════════════════════════════════════════════════════════

# # # def run_pio(val_ds: ERCOTDataset, max_steps: int = 2000) -> dict:
# # #     """Perfect information LP — theoretical upper bound (reference only)."""
# # #     try:
# # #         from scipy.optimize import linprog
# # #     except ImportError:
# # #         print("  [PIO] scipy not installed.")
# # #         return {"name": "PIO (skip)", "cash_revenue": float("nan"),
# # #                 "charge_frac": 0.0, "initial_soc": 0.5, "final_soc": 0.5,
# # #                 "n_steps": 0}

# # #     T      = min(max_steps, len(val_ds) - WINDOW_LEN - 1)
# # #     prices = val_ds.df[PRICE_COLS[0]].iloc[WINDOW_LEN:WINDOW_LEN + T].values

# # #     rev_per_mw = prices * INTERVAL_H
# # #     c      = np.concatenate([-rev_per_mw * (-1), -rev_per_mw * (1)])
# # #     bounds = [(0, BATTERY_POWER_MW)] * T + [(0, BATTERY_POWER_MW)] * T

# # #     coeff = INTERVAL_H / BATTERY_CAP_MWH
# # #     A_ub  = []
# # #     for t in range(T):
# # #         row = np.zeros(2 * T)
# # #         row[:t+1]    =  coeff
# # #         row[T:T+t+1] = -coeff
# # #         A_ub.append(row)
# # #         A_ub.append(-row)
# # #     b_ub = np.array([SOC_MAX - 0.5] * T + [-(SOC_MIN - 0.5)] * T)

# # #     result = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub,
# # #                      bounds=bounds, method="highs")

# # #     if result.success:
# # #         charge    = result.x[:T]
# # #         discharge = result.x[T:]
# # #         revenues  = prices * (discharge - charge) * INTERVAL_H
# # #         total_rev = float(revenues.sum())
# # #     else:
# # #         print(f"  [PIO] LP failed: {result.message}")
# # #         total_rev = float("nan")

# # #     return {
# # #         "name":         "PIO (perfect foresight)",
# # #         "cash_revenue": total_rev,
# # #         "charge_frac":  float(np.mean(charge > 0.1)) if result.success else 0.0,
# # #         "initial_soc":  0.5,
# # #         "final_soc":    0.5,
# # #         "n_steps":      T,
# # #     }


# # # # ════════════════════════════════════════════════════════
# # # # TRAINED SAC AGENT
# # # # ════════════════════════════════════════════════════════

# # # def run_sac_agent(val_ds: ERCOTDataset, max_steps: int = 18000) -> dict:
# # #     """Deterministic SAC rollout. Loads best healthy checkpoint."""

# # #     # Try checkpoints in order of preference
# # #     for ckpt_name in ["stage1_best.pt", "stage1_emergency.pt",
# # #                       "stage1_step50000.pt", "stage1_final.pt"]:
# # #         ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
# # #         if os.path.exists(ckpt_path):
# # #             # Check WINDOW_LEN compatibility
# # #             ckpt = torch.load(ckpt_path, map_location="cpu")
# # #             pos_enc_shape = ckpt["ttfe"]["pos_enc"].shape
# # #             if pos_enc_shape[1] != WINDOW_LEN:
# # #                 print(f"  [SAC] Skip {ckpt_name}: "
# # #                       f"WINDOW_LEN mismatch ({pos_enc_shape[1]} vs {WINDOW_LEN})")
# # #                 continue
# # #             print(f"  [SAC] Using checkpoint: {ckpt_name}")
# # #             best_ckpt = ckpt_path
# # #             break
# # #     else:
# # #         print("  [SAC] No compatible checkpoint found.")
# # #         return {"name": "SAC (no ckpt)", "cash_revenue": float("nan"),
# # #                 "charge_frac": 0.0, "initial_soc": 0.5, "final_soc": 0.5,
# # #                 "n_steps": 0}

# # #     agent = SACAgent()
# # #     step  = agent.load(best_ckpt)
# # #     print(f"  [SAC] Loaded from step {step}")

# # #     env = ERCOTEnv(val_ds)
# # #     obs = env.reset()
# # #     pw, sv, tf, soc_arr = obs
# # #     soc_val      = float(soc_arr[0])
# # #     initial_soc  = soc_val

# # #     total_cash   = 0.0
# # #     charge_count = 0
# # #     n_steps      = 0

# # #     while n_steps < max_steps:
# # #         action, new_soc = agent.select_action(pw, sv, tf, soc_val, deterministic=True)
# # #         (pw, sv, tf, soc_arr), cash_reward, done = env.step(action, new_soc)
# # #         soc_val = float(soc_arr[0])

# # #         total_cash += cash_reward
# # #         if action > 0:
# # #             charge_count += 1
# # #         n_steps += 1

# # #         if done:
# # #             break

# # #     return {
# # #         "name":         f"SAC agent (step {step})",
# # #         "cash_revenue": total_cash,
# # #         "charge_frac":  charge_count / max(n_steps, 1),
# # #         "initial_soc":  initial_soc,
# # #         "final_soc":    soc_val,
# # #         "n_steps":      n_steps,
# # #     }


# # # # ════════════════════════════════════════════════════════
# # # # PRINT RESULTS
# # # # ════════════════════════════════════════════════════════

# # # def print_results(results: list):
# # #     # Use EMA heuristic as comparison baseline (fair, no look-ahead)
# # #     ema_h = next((r for r in results if "EMA" in r["name"]), None)
# # #     ema_rev = ema_h["cash_revenue"] if ema_h else 0.0

# # #     print("\n" + "=" * 75)
# # #     print("Evaluation Results (Cash Revenue Only)")
# # #     print("=" * 75)
# # #     print(f"  {'Method':<40} {'Cash ($)':>12} {'vs EMA Heur':>12} {'Charge%':>8}")
# # #     print(f"  {'-'*75}")

# # #     for r in results:
# # #         rev  = r["cash_revenue"]
# # #         cfrac= r["charge_frac"]
# # #         if np.isnan(rev):
# # #             print(f"  {r['name']:<40} {'n/a':>12} {'n/a':>12} {'n/a':>8}")
# # #         else:
# # #             vs_ema = f"{(rev/ema_rev - 1)*100:>+.1f}%" if ema_rev else "n/a"
# # #             print(f"  {r['name']:<40} ${rev:>10,.2f} {vs_ema:>12} "
# # #                   f"{cfrac*100:>7.1f}%  (SoC: {r['initial_soc']:.2f}→{r['final_soc']:.2f})")

# # #     print()
# # #     print("  Notes:")
# # #     print("  - EMA heuristic: same adaptive threshold as training (fair)")
# # #     print("  - Global median heuristic: uses 5-year look-ahead (unfair advantage)")
# # #     print("  - PIO: 2000 steps only, perfect future knowledge (reference ceiling)")
# # #     print("  - Charge% shows whether agent learned to charge at low prices")
# # #     print("    Healthy: 30-60%.  Collapsed: < 5%.")


# # # # ════════════════════════════════════════════════════════
# # # # MAIN
# # # # ════════════════════════════════════════════════════════

# # # def main():
# # #     print("=" * 60)
# # #     print("Pipeline 5 — Evaluation (Plan D)")
# # #     print("=" * 60)

# # #     val_ds = ERCOTDataset("val")

# # #     print("\n[1] Running EMA heuristic (fair baseline)...")
# # #     ema_rule = run_rule_based(val_ds, max_steps=18000)
# # #     print(f"    Cash: ${ema_rule['cash_revenue']:,.2f} | "
# # #           f"Charge%: {ema_rule['charge_frac']*100:.1f}% | "
# # #           f"SoC: {ema_rule['initial_soc']:.2f}→{ema_rule['final_soc']:.2f}")

# # #     print("\n[2] Running global median heuristic (look-ahead, for reference)...")
# # #     med_rule = run_global_median_heuristic(val_ds, max_steps=18000)
# # #     print(f"    Cash: ${med_rule['cash_revenue']:,.2f} | "
# # #           f"Charge%: {med_rule['charge_frac']*100:.1f}% | "
# # #           f"SoC: {med_rule['initial_soc']:.2f}→{med_rule['final_soc']:.2f}")

# # #     print("\n[3] Running PIO upper bound (2000 steps, reference only)...")
# # #     pio = run_pio(val_ds, max_steps=2000)
# # #     print(f"    Cash: ${pio['cash_revenue']:,.2f}")

# # #     print("\n[4] Running trained SAC agent (deterministic)...")
# # #     sac = run_sac_agent(val_ds, max_steps=18000)
# # #     print(f"    Cash: ${sac['cash_revenue']:,.2f} | "
# # #           f"Charge%: {sac['charge_frac']*100:.1f}% | "
# # #           f"SoC: {sac['initial_soc']:.2f}→{sac['final_soc']:.2f}")

# # #     results = [ema_rule, med_rule, pio, sac]
# # #     print_results(results)

# # #     os.makedirs(LOG_DIR, exist_ok=True)
# # #     out = pd.DataFrame([{
# # #         "name":         r["name"],
# # #         "cash_revenue": r["cash_revenue"],
# # #         "charge_frac":  r["charge_frac"],
# # #         "final_soc":    r["final_soc"],
# # #         "n_steps":      r["n_steps"],
# # #     } for r in results])
# # #     out.to_csv(os.path.join(LOG_DIR, "eval_results.csv"), index=False)
# # #     print(f"\n  Results saved → {LOG_DIR}/eval_results.csv")
# # #     print("\n" + "=" * 60)
# # #     print("✓ Evaluation complete.")


# # # if __name__ == "__main__":
# # #     main()

# # """
# # Pipeline 5 — Evaluation for Plan C
# # ==================================
# # Evaluates the best Plan C checkpoint using:

# # 1. Rule-based training-median heuristic
# # 2. PIO perfect-foresight upper bound
# # 3. SAC Plan C best checkpoint

# # Important:
# # - No EMA reward
# # - No TAU_S
# # - Uses training median p_ref = $24.21/MWh
# # - Reports cash revenue and inventory-adjusted profit
# # - Evaluates stage1_best.pt, not final/emergency checkpoint
# # """

# # import os
# # import sys
# # import numpy as np
# # import pandas as pd
# # import torch

# # sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# # from pipeline.config import *
# # from pipeline.p3_models import FeasibilityProjection
# # from pipeline.p4_train import ERCOTDataset, ERCOTEnv, SACAgent


# # # ════════════════════════════════════════════════════════
# # # Helpers
# # # ════════════════════════════════════════════════════════

# # def make_env(dataset: ERCOTDataset, p_ref: float) -> ERCOTEnv:
# #     """
# #     Create env with training p_ref.
# #     Handles both constructor styles safely.
# #     """
# #     try:
# #         env = ERCOTEnv(dataset, p_ref=p_ref)
# #     except TypeError:
# #         env = ERCOTEnv(dataset)
# #         env.p_ref = p_ref
# #     return env


# # def deterministic_start(env: ERCOTEnv):
# #     env.idx = WINDOW_LEN
# #     env.soc = 0.5
# #     env.ep_steps = 0
# #     return env._obs()


# # def eval_step_cash_only(env: ERCOTEnv, action: float, new_soc: float):
# #     """
# #     Evaluation step:
# #     - cash reward only
# #     - no training reward shaping
# #     - no 288-step episode limit
# #     """
# #     rt_lmp = env.ds.get_rt_lmp(env.idx)

# #     grid_mwh = -action * BATTERY_POWER_MW * INTERVAL_H
# #     cash_reward = grid_mwh * rt_lmp
# #     degradation = CYCLE_COST_PER_MWH * abs(grid_mwh)

# #     env.soc = float(np.clip(new_soc, SOC_MIN, SOC_MAX))
# #     env.idx += 1
# #     env.ep_steps += 1

# #     done = env.idx >= len(env.ds) - 1
# #     return env._obs(), cash_reward, done, degradation


# # def inventory_adjusted_profit(
# #     cash_revenue: float,
# #     initial_soc: float,
# #     final_soc: float,
# #     degradation_cost: float,
# #     p_ref: float,
# # ) -> float:
# #     inventory_change = (final_soc - initial_soc) * BATTERY_CAP_MWH * p_ref
# #     return cash_revenue + inventory_change - degradation_cost


# # def project_action(raw_action: float, soc: float, proj: FeasibilityProjection):
# #     raw_t = torch.tensor([[raw_action]], dtype=torch.float32, device=DEVICE)
# #     soc_t = torch.tensor([[soc]], dtype=torch.float32, device=DEVICE)

# #     with torch.no_grad():
# #         feasible_t, new_soc_t = proj(raw_t, soc_t)

# #     return float(feasible_t.item()), float(new_soc_t.item())


# # # ════════════════════════════════════════════════════════
# # # Baseline 1 — Training Median Heuristic
# # # ════════════════════════════════════════════════════════

# # def run_median_heuristic(val_ds: ERCOTDataset, p_ref: float, max_steps: int = 18000):
# #     """
# #     Rule:
# #         charge if price < training median p_ref
# #         discharge if price >= training median p_ref
# #     """
# #     env = make_env(val_ds, p_ref=p_ref)
# #     obs = deterministic_start(env)

# #     pw, sv, tf, soc_arr = obs
# #     soc = float(soc_arr[0])
# #     initial_soc = soc

# #     proj = FeasibilityProjection().to(DEVICE)

# #     total_cash = 0.0
# #     total_deg = 0.0
# #     charge_count = 0
# #     discharge_count = 0
# #     hold_count = 0
# #     n_steps = 0

# #     while n_steps < max_steps:
# #         rt_lmp = val_ds.get_rt_lmp(env.idx)

# #         raw_action = 1.0 if rt_lmp < p_ref else -1.0
# #         action, new_soc = project_action(raw_action, soc, proj)

# #         (pw, sv, tf, soc_arr), cash, done, deg = eval_step_cash_only(
# #             env, action, new_soc
# #         )
# #         soc = float(soc_arr[0])

# #         total_cash += cash
# #         total_deg += deg

# #         if action > 1e-6:
# #             charge_count += 1
# #         elif action < -1e-6:
# #             discharge_count += 1
# #         else:
# #             hold_count += 1

# #         n_steps += 1
# #         if done:
# #             break

# #     final_soc = soc
# #     inv_adj = inventory_adjusted_profit(
# #         total_cash, initial_soc, final_soc, total_deg, p_ref
# #     )

# #     return {
# #         "name": f"Median heuristic p_ref=${p_ref:.2f}",
# #         "cash_revenue": total_cash,
# #         "inv_adjusted": inv_adj,
# #         "degradation": total_deg,
# #         "initial_soc": initial_soc,
# #         "final_soc": final_soc,
# #         "charge_frac": charge_count / max(n_steps, 1),
# #         "discharge_frac": discharge_count / max(n_steps, 1),
# #         "hold_frac": hold_count / max(n_steps, 1),
# #         "n_steps": n_steps,
# #     }


# # # ════════════════════════════════════════════════════════
# # # Baseline 2 — PIO
# # # ════════════════════════════════════════════════════════

# # def run_pio(val_ds: ERCOTDataset, p_ref: float, max_steps: int = 2000):
# #     """
# #     Perfect information LP upper bound.
# #     This is only a reference ceiling.
# #     """
# #     try:
# #         from scipy.optimize import linprog
# #     except ImportError:
# #         print("[PIO] scipy not installed. Skipping PIO.")
# #         return {
# #             "name": "PIO skipped",
# #             "cash_revenue": float("nan"),
# #             "inv_adjusted": float("nan"),
# #             "degradation": 0.0,
# #             "initial_soc": 0.5,
# #             "final_soc": 0.5,
# #             "charge_frac": 0.0,
# #             "discharge_frac": 0.0,
# #             "hold_frac": 0.0,
# #             "n_steps": 0,
# #         }

# #     T = min(max_steps, len(val_ds) - WINDOW_LEN - 1)
# #     prices = val_ds.df[PRICE_COLS[0]].iloc[WINDOW_LEN:WINDOW_LEN + T].values

# #     rev_per_mw = prices * INTERVAL_H

# #     # Variables: [charge_0...charge_T, discharge_0...discharge_T]
# #     # Minimize negative profit
# #     c = np.concatenate([
# #         rev_per_mw,      # charging cost
# #         -rev_per_mw,     # discharging revenue
# #     ])

# #     bounds = [(0, BATTERY_POWER_MW)] * T + [(0, BATTERY_POWER_MW)] * T

# #     coeff = INTERVAL_H / BATTERY_CAP_MWH
# #     A_ub = []
# #     b_ub = []

# #     for t in range(T):
# #         row = np.zeros(2 * T)
# #         row[:t + 1] = coeff
# #         row[T:T + t + 1] = -coeff

# #         # SoC upper bound
# #         A_ub.append(row)
# #         b_ub.append(SOC_MAX - 0.5)

# #         # SoC lower bound
# #         A_ub.append(-row)
# #         b_ub.append(-(SOC_MIN - 0.5))

# #     result = linprog(
# #         c,
# #         A_ub=np.array(A_ub),
# #         b_ub=np.array(b_ub),
# #         bounds=bounds,
# #         method="highs",
# #     )

# #     if not result.success:
# #         print(f"[PIO] LP failed: {result.message}")
# #         total_cash = float("nan")
# #         final_soc = 0.5
# #         charge_frac = 0.0
# #         discharge_frac = 0.0
# #     else:
# #         charge = result.x[:T]
# #         discharge = result.x[T:]

# #         revenues = prices * (discharge - charge) * INTERVAL_H
# #         total_cash = float(revenues.sum())

# #         final_soc = 0.5 + float((charge.sum() - discharge.sum()) * INTERVAL_H / BATTERY_CAP_MWH)

# #         charge_frac = float(np.mean(charge > 1e-6))
# #         discharge_frac = float(np.mean(discharge > 1e-6))

# #     inv_adj = inventory_adjusted_profit(
# #         total_cash,
# #         0.5,
# #         final_soc,
# #         0.0,
# #         p_ref,
# #     )

# #     return {
# #         "name": f"PIO perfect foresight ({T} steps)",
# #         "cash_revenue": total_cash,
# #         "inv_adjusted": inv_adj,
# #         "degradation": 0.0,
# #         "initial_soc": 0.5,
# #         "final_soc": final_soc,
# #         "charge_frac": charge_frac,
# #         "discharge_frac": discharge_frac,
# #         "hold_frac": 1.0 - charge_frac - discharge_frac,
# #         "n_steps": T,
# #     }


# # # ════════════════════════════════════════════════════════
# # # SAC Agent Evaluation
# # # ════════════════════════════════════════════════════════

# # def load_best_agent():
# #     ckpt_path = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")

# #     if not os.path.exists(ckpt_path):
# #         raise FileNotFoundError(
# #             f"Could not find {ckpt_path}. "
# #             "You should evaluate stage1_best.pt, not final/emergency."
# #         )

# #     agent = SACAgent()

# #     try:
# #         step = agent.load(ckpt_path)
# #     except Exception:
# #         ckpt = torch.load(ckpt_path, map_location=DEVICE)
# #         agent.ttfe.load_state_dict(ckpt["ttfe"])
# #         agent.actor.load_state_dict(ckpt["actor"])
# #         agent.critic.load_state_dict(ckpt["critic"])

# #         if "critic_tgt" in ckpt:
# #             agent.critic_tgt.load_state_dict(ckpt["critic_tgt"])
# #         else:
# #             agent.critic_tgt.load_state_dict(agent.critic.state_dict())

# #         step = ckpt.get("step", -1)

# #     print(f"[SAC] Loaded best checkpoint: {ckpt_path} | step={step}")
# #     return agent, step


# # def run_sac_agent(val_ds: ERCOTDataset, p_ref: float, max_steps: int = 18000):
# #     agent, step = load_best_agent()

# #     env = make_env(val_ds, p_ref=p_ref)
# #     obs = deterministic_start(env)

# #     pw, sv, tf, soc_arr = obs
# #     soc = float(soc_arr[0])
# #     initial_soc = soc

# #     total_cash = 0.0
# #     total_deg = 0.0
# #     charge_count = 0
# #     discharge_count = 0
# #     hold_count = 0
# #     n_steps = 0

# #     while n_steps < max_steps:
# #         action, new_soc = agent.select_action(
# #             pw, sv, tf, soc, deterministic=True
# #         )

# #         (pw, sv, tf, soc_arr), cash, done, deg = eval_step_cash_only(
# #             env, action, new_soc
# #         )
# #         soc = float(soc_arr[0])

# #         total_cash += cash
# #         total_deg += deg

# #         if action > 1e-6:
# #             charge_count += 1
# #         elif action < -1e-6:
# #             discharge_count += 1
# #         else:
# #             hold_count += 1

# #         n_steps += 1
# #         if done:
# #             break

# #     final_soc = soc
# #     inv_adj = inventory_adjusted_profit(
# #         total_cash, initial_soc, final_soc, total_deg, p_ref
# #     )

# #     return {
# #         "name": f"SAC Plan C best checkpoint step {step}",
# #         "cash_revenue": total_cash,
# #         "inv_adjusted": inv_adj,
# #         "degradation": total_deg,
# #         "initial_soc": initial_soc,
# #         "final_soc": final_soc,
# #         "charge_frac": charge_count / max(n_steps, 1),
# #         "discharge_frac": discharge_count / max(n_steps, 1),
# #         "hold_frac": hold_count / max(n_steps, 1),
# #         "n_steps": n_steps,
# #     }


# # # ════════════════════════════════════════════════════════
# # # Results
# # # ════════════════════════════════════════════════════════

# # def print_results(results):
# #     heuristic = next(
# #         (r for r in results if "Median heuristic" in r["name"]),
# #         None,
# #     )
# #     heuristic_inv = heuristic["inv_adjusted"] if heuristic else None

# #     print("\n" + "=" * 100)
# #     print("Plan C Evaluation Results")
# #     print("=" * 100)
# #     print(
# #         f"{'Method':<42} "
# #         f"{'Cash($)':>12} "
# #         f"{'InvAdj($)':>12} "
# #         f"{'vs Heur':>10} "
# #         f"{'FinalSoC':>9} "
# #         f"{'Charge%':>8} "
# #         f"{'Hold%':>8} "
# #         f"{'Steps':>7}"
# #     )
# #     print("-" * 100)

# #     for r in results:
# #         inv = r["inv_adjusted"]
# #         vs = "n/a"

# #         if heuristic_inv is not None and heuristic_inv != 0 and not np.isnan(inv):
# #             vs = f"{(inv / heuristic_inv - 1.0) * 100:+.1f}%"

# #         print(
# #             f"{r['name']:<42} "
# #             f"${r['cash_revenue']:>11,.2f} "
# #             f"${r['inv_adjusted']:>11,.2f} "
# #             f"{vs:>10} "
# #             f"{r['final_soc']:>9.3f} "
# #             f"{r['charge_frac'] * 100:>7.1f}% "
# #             f"{r['hold_frac'] * 100:>7.1f}% "
# #             f"{r['n_steps']:>7}"
# #         )

# #     print("=" * 100)


# # def main():
# #     print("=" * 60)
# #     print("Pipeline 5 — Evaluation for Plan C")
# #     print("=" * 60)

# #     train_ds = ERCOTDataset("train")
# #     val_ds = ERCOTDataset("val")

# #     p_ref = float(train_ds.df[PRICE_COLS[0]].median())
# #     print(f"[Eval] Training p_ref = ${p_ref:.2f}/MWh")
# #     print("[Eval] No EMA. No TAU_S. Evaluating Plan C.\n")

# #     print("[1] Running training-median heuristic...")
# #     heuristic = run_median_heuristic(val_ds, p_ref=p_ref, max_steps=18000)

# #     print("[2] Running PIO reference...")
# #     pio = run_pio(val_ds, p_ref=p_ref, max_steps=2000)

# #     print("[3] Running SAC Plan C best checkpoint...")
# #     sac = run_sac_agent(val_ds, p_ref=p_ref, max_steps=18000)

# #     results = [heuristic, pio, sac]
# #     print_results(results)

# #     os.makedirs(LOG_DIR, exist_ok=True)
# #     out_path = os.path.join(LOG_DIR, "eval_results_planc.csv")
# #     pd.DataFrame(results).to_csv(out_path, index=False)
# #     print(f"\nSaved results → {out_path}")


# # if __name__ == "__main__":
# #     main()
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# from pipeline.config import *
# from pipeline.p3_models import FeasibilityProjection
# from pipeline.p4_train import ERCOTDataset, ERCOTEnv, SACAgent


# # ════════════════════════════════════════════════════════
# # Helpers
# # ════════════════════════════════════════════════════════

# def make_env(dataset: ERCOTDataset, p_ref: float) -> ERCOTEnv:
#     """
#     Create env with training p_ref.
#     Handles both constructor styles safely.
#     """
#     try:
#         env = ERCOTEnv(dataset, p_ref=p_ref)
#     except TypeError:
#         env = ERCOTEnv(dataset)
#         env.p_ref = p_ref
#     return env


# def deterministic_start(env: ERCOTEnv):
#     env.idx = WINDOW_LEN
#     env.soc = 0.5
#     env.ep_steps = 0
#     return env._obs()


# def eval_step_cash_only(env: ERCOTEnv, action: float, new_soc: float):
#     """
#     Evaluation step:
#     - cash reward only
#     - no training reward shaping
#     - no 288-step episode limit
#     """
#     rt_lmp = env.ds.get_rt_lmp(env.idx)

#     grid_mwh = -action * BATTERY_POWER_MW * INTERVAL_H
#     cash_reward = grid_mwh * rt_lmp
#     degradation = CYCLE_COST_PER_MWH * abs(grid_mwh)

#     env.soc = float(np.clip(new_soc, SOC_MIN, SOC_MAX))
#     env.idx += 1
#     env.ep_steps += 1

#     done = env.idx >= len(env.ds) - 1
#     return env._obs(), cash_reward, done, degradation


# def inventory_adjusted_profit(
#     cash_revenue: float,
#     initial_soc: float,
#     final_soc: float,
#     degradation_cost: float,
#     p_ref: float,
# ) -> float:
#     inventory_change = (final_soc - initial_soc) * BATTERY_CAP_MWH * p_ref
#     return cash_revenue + inventory_change - degradation_cost


# def project_action(raw_action: float, soc: float, proj: FeasibilityProjection):
#     raw_t = torch.tensor([[raw_action]], dtype=torch.float32, device=DEVICE)
#     soc_t = torch.tensor([[soc]], dtype=torch.float32, device=DEVICE)

#     with torch.no_grad():
#         feasible_t, new_soc_t = proj(raw_t, soc_t)

#     return float(feasible_t.item()), float(new_soc_t.item())


# # ════════════════════════════════════════════════════════
# # Baseline 1 — Training Median Heuristic
# # ════════════════════════════════════════════════════════

# def run_median_heuristic(val_ds: ERCOTDataset, p_ref: float, max_steps: int = 18000):
#     """
#     Rule:
#         charge if price < training median p_ref
#         discharge if price >= training median p_ref
#     """
#     env = make_env(val_ds, p_ref=p_ref)
#     obs = deterministic_start(env)

#     pw, sv, tf, soc_arr = obs
#     soc = float(soc_arr[0])
#     initial_soc = soc

#     proj = FeasibilityProjection().to(DEVICE)

#     total_cash = 0.0
#     total_deg = 0.0
#     charge_count = 0
#     discharge_count = 0
#     hold_count = 0
#     n_steps = 0

#     while n_steps < max_steps:
#         rt_lmp = val_ds.get_rt_lmp(env.idx)

#         raw_action = 1.0 if rt_lmp < p_ref else -1.0
#         action, new_soc = project_action(raw_action, soc, proj)

#         (pw, sv, tf, soc_arr), cash, done, deg = eval_step_cash_only(
#             env, action, new_soc
#         )
#         soc = float(soc_arr[0])

#         total_cash += cash
#         total_deg += deg

#         if action > 1e-6:
#             charge_count += 1
#         elif action < -1e-6:
#             discharge_count += 1
#         else:
#             hold_count += 1

#         n_steps += 1
#         if done:
#             break

#     final_soc = soc
#     inv_adj = inventory_adjusted_profit(
#         total_cash, initial_soc, final_soc, total_deg, p_ref
#     )

#     return {
#         "name": f"Median heuristic p_ref=${p_ref:.2f}",
#         "cash_revenue": total_cash,
#         "inv_adjusted": inv_adj,
#         "degradation": total_deg,
#         "initial_soc": initial_soc,
#         "final_soc": final_soc,
#         "charge_frac": charge_count / max(n_steps, 1),
#         "discharge_frac": discharge_count / max(n_steps, 1),
#         "hold_frac": hold_count / max(n_steps, 1),
#         "n_steps": n_steps,
#     }


# # ════════════════════════════════════════════════════════
# # Baseline 2 — PIO
# # ════════════════════════════════════════════════════════

# def run_pio(val_ds: ERCOTDataset, p_ref: float, max_steps: int = 2000):
#     """
#     Perfect information LP upper bound.
#     This is only a reference ceiling.
#     """
#     try:
#         from scipy.optimize import linprog
#     except ImportError:
#         print("[PIO] scipy not installed. Skipping PIO.")
#         return {
#             "name": "PIO skipped",
#             "cash_revenue": float("nan"),
#             "inv_adjusted": float("nan"),
#             "degradation": 0.0,
#             "initial_soc": 0.5,
#             "final_soc": 0.5,
#             "charge_frac": 0.0,
#             "discharge_frac": 0.0,
#             "hold_frac": 0.0,
#             "n_steps": 0,
#         }

#     T = min(max_steps, len(val_ds) - WINDOW_LEN - 1)
#     prices = val_ds.df[PRICE_COLS[0]].iloc[WINDOW_LEN:WINDOW_LEN + T].values

#     rev_per_mw = prices * INTERVAL_H

#     # Variables: [charge_0...charge_T, discharge_0...discharge_T]
#     # Minimize negative profit
#     c = np.concatenate([
#         rev_per_mw,      # charging cost
#         -rev_per_mw,     # discharging revenue
#     ])

#     bounds = [(0, BATTERY_POWER_MW)] * T + [(0, BATTERY_POWER_MW)] * T

#     coeff = INTERVAL_H / BATTERY_CAP_MWH
#     A_ub = []
#     b_ub = []

#     for t in range(T):
#         row = np.zeros(2 * T)
#         row[:t + 1] = coeff
#         row[T:T + t + 1] = -coeff

#         # SoC upper bound
#         A_ub.append(row)
#         b_ub.append(SOC_MAX - 0.5)

#         # SoC lower bound
#         A_ub.append(-row)
#         b_ub.append(-(SOC_MIN - 0.5))

#     result = linprog(
#         c,
#         A_ub=np.array(A_ub),
#         b_ub=np.array(b_ub),
#         bounds=bounds,
#         method="highs",
#     )

#     if not result.success:
#         print(f"[PIO] LP failed: {result.message}")
#         total_cash = float("nan")
#         final_soc = 0.5
#         charge_frac = 0.0
#         discharge_frac = 0.0
#     else:
#         charge = result.x[:T]
#         discharge = result.x[T:]

#         revenues = prices * (discharge - charge) * INTERVAL_H
#         total_cash = float(revenues.sum())

#         final_soc = 0.5 + float((charge.sum() - discharge.sum()) * INTERVAL_H / BATTERY_CAP_MWH)

#         charge_frac = float(np.mean(charge > 1e-6))
#         discharge_frac = float(np.mean(discharge > 1e-6))

#     inv_adj = inventory_adjusted_profit(
#         total_cash,
#         0.5,
#         final_soc,
#         0.0,
#         p_ref,
#     )

#     return {
#         "name": f"PIO perfect foresight ({T} steps)",
#         "cash_revenue": total_cash,
#         "inv_adjusted": inv_adj,
#         "degradation": 0.0,
#         "initial_soc": 0.5,
#         "final_soc": final_soc,
#         "charge_frac": charge_frac,
#         "discharge_frac": discharge_frac,
#         "hold_frac": 1.0 - charge_frac - discharge_frac,
#         "n_steps": T,
#     }


# # ════════════════════════════════════════════════════════
# # SAC Agent Evaluation
# # ════════════════════════════════════════════════════════

# def load_best_agent():
#     ckpt_path = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")

#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(
#             f"Could not find {ckpt_path}. "
#             "You should evaluate stage1_best.pt, not final/emergency."
#         )

#     agent = SACAgent()

#     try:
#         step = agent.load(ckpt_path)
#     except Exception:
#         ckpt = torch.load(ckpt_path, map_location=DEVICE)
#         agent.ttfe.load_state_dict(ckpt["ttfe"])
#         agent.actor.load_state_dict(ckpt["actor"])
#         agent.critic.load_state_dict(ckpt["critic"])

#         if "critic_tgt" in ckpt:
#             agent.critic_tgt.load_state_dict(ckpt["critic_tgt"])
#         else:
#             agent.critic_tgt.load_state_dict(agent.critic.state_dict())

#         step = ckpt.get("step", -1)

#     print(f"[SAC] Loaded best checkpoint: {ckpt_path} | step={step}")
#     return agent, step


# def run_sac_agent(val_ds: ERCOTDataset, p_ref: float, max_steps: int = 18000):
#     agent, step = load_best_agent()

#     env = make_env(val_ds, p_ref=p_ref)
#     obs = deterministic_start(env)

#     pw, sv, tf, soc_arr = obs
#     soc = float(soc_arr[0])
#     initial_soc = soc

#     total_cash = 0.0
#     total_deg = 0.0
#     charge_count = 0
#     discharge_count = 0
#     hold_count = 0
#     n_steps = 0

#     while n_steps < max_steps:
#         action, new_soc = agent.select_action(
#             pw, sv, tf, soc, deterministic=True
#         )

#         (pw, sv, tf, soc_arr), cash, done, deg = eval_step_cash_only(
#             env, action, new_soc
#         )
#         soc = float(soc_arr[0])

#         total_cash += cash
#         total_deg += deg

#         if action > 1e-6:
#             charge_count += 1
#         elif action < -1e-6:
#             discharge_count += 1
#         else:
#             hold_count += 1

#         n_steps += 1
#         if done:
#             break

#     final_soc = soc
#     inv_adj = inventory_adjusted_profit(
#         total_cash, initial_soc, final_soc, total_deg, p_ref
#     )

#     return {
#         "name": f"SAC Plan C best checkpoint step {step}",
#         "cash_revenue": total_cash,
#         "inv_adjusted": inv_adj,
#         "degradation": total_deg,
#         "initial_soc": initial_soc,
#         "final_soc": final_soc,
#         "charge_frac": charge_count / max(n_steps, 1),
#         "discharge_frac": discharge_count / max(n_steps, 1),
#         "hold_frac": hold_count / max(n_steps, 1),
#         "n_steps": n_steps,
#     }


# # ════════════════════════════════════════════════════════
# # Results
# # ════════════════════════════════════════════════════════

# def print_results(results):
#     heuristic = next(
#         (r for r in results if "Median heuristic" in r["name"]),
#         None,
#     )
#     heuristic_inv = heuristic["inv_adjusted"] if heuristic else None

#     print("\n" + "=" * 100)
#     print("Plan C Evaluation Results  [TEST SET — held out]")
#     print("=" * 100)
#     print(
#         f"{'Method':<42} "
#         f"{'Cash($)':>12} "
#         f"{'InvAdj($)':>12} "
#         f"{'vs Heur':>10} "
#         f"{'FinalSoC':>9} "
#         f"{'Charge%':>8} "
#         f"{'Hold%':>8} "
#         f"{'Steps':>7}"
#     )
#     print("-" * 100)

#     for r in results:
#         inv = r["inv_adjusted"]
#         vs = "n/a"

#         if heuristic_inv is not None and heuristic_inv != 0 and not np.isnan(inv):
#             vs = f"{(inv / heuristic_inv - 1.0) * 100:+.1f}%"

#         print(
#             f"{r['name']:<42} "
#             f"${r['cash_revenue']:>11,.2f} "
#             f"${r['inv_adjusted']:>11,.2f} "
#             f"{vs:>10} "
#             f"{r['final_soc']:>9.3f} "
#             f"{r['charge_frac'] * 100:>7.1f}% "
#             f"{r['hold_frac'] * 100:>7.1f}% "
#             f"{r['n_steps']:>7}"
#         )

#     print("=" * 100)


# def main():
#     print("=" * 60)
#     print("Pipeline 5 — Evaluation for Plan C  [TEST SET]")
#     print("=" * 60)

#     train_ds = ERCOTDataset("train")

#     # ── CRITICAL: Use TEST split, NOT val split ──────────────────────
#     # stage1_best.pt was selected based on val performance during training.
#     # Evaluating on val split = data leakage (same data selected the checkpoint).
#     # Test split [TEST_START, STAGE1_END] was never seen during training or
#     # checkpoint selection. This is the only valid split for final results.
#     test_ds = ERCOTDataset("test")

#     p_ref = float(train_ds.df[PRICE_COLS[0]].median())
#     print(f"[Eval] Training p_ref = ${p_ref:.2f}/MWh")
#     print(f"[Eval] TEST split     = {TEST_START} → {STAGE1_END}")
#     print(f"[Eval] Val split ({VAL_START}→{TEST_START}) was used for")
#     print(f"[Eval] checkpoint selection only — NOT reported here.\n")

#     print("[1] Running training-median heuristic on TEST set...")
#     heuristic = run_median_heuristic(test_ds, p_ref=p_ref, max_steps=18000)

#     print("[2] Running PIO reference on TEST set...")
#     pio = run_pio(test_ds, p_ref=p_ref, max_steps=2000)

#     print("[3] Running SAC Plan C best checkpoint on TEST set...")
#     sac = run_sac_agent(test_ds, p_ref=p_ref, max_steps=18000)

#     results = [heuristic, pio, sac]
#     print_results(results)

#     os.makedirs(LOG_DIR, exist_ok=True)
#     out_path = os.path.join(LOG_DIR, "eval_results_planc_test.csv")
#     pd.DataFrame(results).to_csv(out_path, index=False)
#     print(f"\nSaved results → {out_path}")
#     print("NOTE: These results use the TEST split — final reportable numbers.")


# if __name__ == "__main__":
#     main()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.config import *
from pipeline.p3_models import FeasibilityProjection
from pipeline.p4_train import ERCOTDataset, ERCOTEnv, SACAgent


# ════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════

def make_env(dataset: ERCOTDataset, p_ref: float) -> ERCOTEnv:
    """
    Create env with training p_ref.
    Handles both constructor styles safely.
    """
    try:
        env = ERCOTEnv(dataset, p_ref=p_ref)
    except TypeError:
        env = ERCOTEnv(dataset)
        env.p_ref = p_ref
    return env


def deterministic_start(env: ERCOTEnv):
    env.idx = WINDOW_LEN
    env.soc = 0.5
    env.ep_steps = 0
    return env._obs()


def eval_step_cash_only(env: ERCOTEnv, action: float, new_soc: float):
    """
    Evaluation step:
    - cash reward only
    - no training reward shaping
    - no 288-step episode limit
    """
    rt_lmp = env.ds.get_rt_lmp(env.idx)

    grid_mwh = -action * BATTERY_POWER_MW * INTERVAL_H
    cash_reward = grid_mwh * rt_lmp
    degradation = CYCLE_COST_PER_MWH * abs(grid_mwh)

    env.soc = float(np.clip(new_soc, SOC_MIN, SOC_MAX))
    env.idx += 1
    env.ep_steps += 1

    done = env.idx >= len(env.ds) - 1
    return env._obs(), cash_reward, done, degradation


def inventory_adjusted_profit(
    cash_revenue: float,
    initial_soc: float,
    final_soc: float,
    degradation_cost: float,
    p_ref: float,
) -> float:
    inventory_change = (final_soc - initial_soc) * BATTERY_CAP_MWH * p_ref
    return cash_revenue + inventory_change - degradation_cost


def project_action(raw_action: float, soc: float, proj: FeasibilityProjection):
    raw_t = torch.tensor([[raw_action]], dtype=torch.float32, device=DEVICE)
    soc_t = torch.tensor([[soc]], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        feasible_t, new_soc_t = proj(raw_t, soc_t)

    return float(feasible_t.item()), float(new_soc_t.item())


# ════════════════════════════════════════════════════════
# Baseline 1 — Training Median Heuristic
# ════════════════════════════════════════════════════════

def run_median_heuristic(val_ds: ERCOTDataset, p_ref: float, max_steps: int = 18000):
    """
    Rule:
        charge if price < training median p_ref
        discharge if price >= training median p_ref
    """
    env = make_env(val_ds, p_ref=p_ref)
    obs = deterministic_start(env)

    pw, sv, tf, soc_arr = obs
    soc = float(soc_arr[0])
    initial_soc = soc

    proj = FeasibilityProjection().to(DEVICE)

    total_cash = 0.0
    total_deg = 0.0
    charge_count = 0
    discharge_count = 0
    hold_count = 0
    n_steps = 0

    while n_steps < max_steps:
        rt_lmp = val_ds.get_rt_lmp(env.idx)

        raw_action = 1.0 if rt_lmp < p_ref else -1.0
        action, new_soc = project_action(raw_action, soc, proj)

        (pw, sv, tf, soc_arr), cash, done, deg = eval_step_cash_only(
            env, action, new_soc
        )
        soc = float(soc_arr[0])

        total_cash += cash
        total_deg += deg

        if action > 1e-6:
            charge_count += 1
        elif action < -1e-6:
            discharge_count += 1
        else:
            hold_count += 1

        n_steps += 1
        if done:
            break

    final_soc = soc
    inv_adj = inventory_adjusted_profit(
        total_cash, initial_soc, final_soc, total_deg, p_ref
    )

    return {
        "name": f"Median heuristic p_ref=${p_ref:.2f}",
        "cash_revenue": total_cash,
        "inv_adjusted": inv_adj,
        "degradation": total_deg,
        "initial_soc": initial_soc,
        "final_soc": final_soc,
        "charge_frac": charge_count / max(n_steps, 1),
        "discharge_frac": discharge_count / max(n_steps, 1),
        "hold_frac": hold_count / max(n_steps, 1),
        "n_steps": n_steps,
    }


# ════════════════════════════════════════════════════════
# Baseline 2 — PIO
# ════════════════════════════════════════════════════════

def run_pio(val_ds: ERCOTDataset, p_ref: float, max_steps: int = 2000):
    """
    Perfect information LP upper bound.
    This is only a reference ceiling.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        print("[PIO] scipy not installed. Skipping PIO.")
        return {
            "name": "PIO skipped",
            "cash_revenue": float("nan"),
            "inv_adjusted": float("nan"),
            "degradation": 0.0,
            "initial_soc": 0.5,
            "final_soc": 0.5,
            "charge_frac": 0.0,
            "discharge_frac": 0.0,
            "hold_frac": 0.0,
            "n_steps": 0,
        }

    T = min(max_steps, len(val_ds) - WINDOW_LEN - 1)
    prices = val_ds.df[PRICE_COLS[0]].iloc[WINDOW_LEN:WINDOW_LEN + T].values

    rev_per_mw = prices * INTERVAL_H

    # Variables: [charge_0...charge_T, discharge_0...discharge_T]
    # Minimize negative profit
    c = np.concatenate([
        rev_per_mw,      # charging cost
        -rev_per_mw,     # discharging revenue
    ])

    bounds = [(0, BATTERY_POWER_MW)] * T + [(0, BATTERY_POWER_MW)] * T

    coeff = INTERVAL_H / BATTERY_CAP_MWH
    A_ub = []
    b_ub = []

    for t in range(T):
        row = np.zeros(2 * T)
        row[:t + 1] = coeff
        row[T:T + t + 1] = -coeff

        # SoC upper bound
        A_ub.append(row)
        b_ub.append(SOC_MAX - 0.5)

        # SoC lower bound
        A_ub.append(-row)
        b_ub.append(-(SOC_MIN - 0.5))

    result = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        print(f"[PIO] LP failed: {result.message}")
        total_cash = float("nan")
        final_soc = 0.5
        charge_frac = 0.0
        discharge_frac = 0.0
    else:
        charge = result.x[:T]
        discharge = result.x[T:]

        revenues = prices * (discharge - charge) * INTERVAL_H
        total_cash = float(revenues.sum())

        final_soc = 0.5 + float((charge.sum() - discharge.sum()) * INTERVAL_H / BATTERY_CAP_MWH)

        charge_frac = float(np.mean(charge > 1e-6))
        discharge_frac = float(np.mean(discharge > 1e-6))

    inv_adj = inventory_adjusted_profit(
        total_cash,
        0.5,
        final_soc,
        0.0,
        p_ref,
    )

    return {
        "name": f"PIO perfect foresight ({T} steps)",
        "cash_revenue": total_cash,
        "inv_adjusted": inv_adj,
        "degradation": 0.0,
        "initial_soc": 0.5,
        "final_soc": final_soc,
        "charge_frac": charge_frac,
        "discharge_frac": discharge_frac,
        "hold_frac": 1.0 - charge_frac - discharge_frac,
        "n_steps": T,
    }


# ════════════════════════════════════════════════════════
# SAC Agent Evaluation
# ════════════════════════════════════════════════════════

def load_best_agent():
    ckpt_path = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Could not find {ckpt_path}. "
            "You should evaluate stage1_best.pt, not final/emergency."
        )

    agent = SACAgent()

    try:
        step = agent.load(ckpt_path)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        agent.ttfe.load_state_dict(ckpt["ttfe"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])

        if "critic_tgt" in ckpt:
            agent.critic_tgt.load_state_dict(ckpt["critic_tgt"])
        else:
            agent.critic_tgt.load_state_dict(agent.critic.state_dict())

        step = ckpt.get("step", -1)

    print(f"[SAC] Loaded best checkpoint: {ckpt_path} | step={step}")
    return agent, step


def run_sac_agent(val_ds: ERCOTDataset, p_ref: float, max_steps: int = 18000):
    agent, step = load_best_agent()

    env = make_env(val_ds, p_ref=p_ref)
    obs = deterministic_start(env)

    pw, sv, tf, soc_arr = obs
    soc = float(soc_arr[0])
    initial_soc = soc

    total_cash = 0.0
    total_deg = 0.0
    charge_count = 0
    discharge_count = 0
    hold_count = 0
    n_steps = 0

    while n_steps < max_steps:
        action, new_soc = agent.select_action(
            pw, sv, tf, soc, deterministic=True
        )

        (pw, sv, tf, soc_arr), cash, done, deg = eval_step_cash_only(
            env, action, new_soc
        )
        soc = float(soc_arr[0])

        total_cash += cash
        total_deg += deg

        if action > 1e-6:
            charge_count += 1
        elif action < -1e-6:
            discharge_count += 1
        else:
            hold_count += 1

        n_steps += 1
        if done:
            break

    final_soc = soc
    inv_adj = inventory_adjusted_profit(
        total_cash, initial_soc, final_soc, total_deg, p_ref
    )

    return {
        "name": f"SAC Plan C best checkpoint step {step}",
        "cash_revenue": total_cash,
        "inv_adjusted": inv_adj,
        "degradation": total_deg,
        "initial_soc": initial_soc,
        "final_soc": final_soc,
        "charge_frac": charge_count / max(n_steps, 1),
        "discharge_frac": discharge_count / max(n_steps, 1),
        "hold_frac": hold_count / max(n_steps, 1),
        "n_steps": n_steps,
    }


# ════════════════════════════════════════════════════════
# Results
# ════════════════════════════════════════════════════════

def print_results(results):
    heuristic = next(
        (r for r in results if "Median heuristic" in r["name"]),
        None,
    )
    heuristic_inv = heuristic["inv_adjusted"] if heuristic else None

    print("\n" + "=" * 100)
    print("Plan C Evaluation Results  [TEST SET — held out]")
    print("=" * 100)
    print(
        f"{'Method':<42} "
        f"{'Cash($)':>12} "
        f"{'InvAdj($)':>12} "
        f"{'vs Heur':>10} "
        f"{'FinalSoC':>9} "
        f"{'Charge%':>8} "
        f"{'Hold%':>8} "
        f"{'Steps':>7}"
    )
    print("-" * 100)

    for r in results:
        inv = r["inv_adjusted"]
        vs = "n/a"

        if heuristic_inv is not None and heuristic_inv != 0 and not np.isnan(inv):
            vs = f"{(inv / heuristic_inv - 1.0) * 100:+.1f}%"

        print(
            f"{r['name']:<42} "
            f"${r['cash_revenue']:>11,.2f} "
            f"${r['inv_adjusted']:>11,.2f} "
            f"{vs:>10} "
            f"{r['final_soc']:>9.3f} "
            f"{r['charge_frac'] * 100:>7.1f}% "
            f"{r['hold_frac'] * 100:>7.1f}% "
            f"{r['n_steps']:>7}"
        )

    print("=" * 100)


def main():
    print("=" * 60)
    print("Pipeline 5 — Evaluation for Plan C  [TEST SET]")
    print("=" * 60)

    train_ds = ERCOTDataset("train")

    # ── CRITICAL: Use TEST split, NOT val split ──────────────────────
    # stage1_best.pt was selected based on val performance during training.
    # Evaluating on val split = data leakage (same data selected the checkpoint).
    # Test split [TEST_START, STAGE1_END] was never seen during training or
    # checkpoint selection. This is the only valid split for final results.
    test_ds = ERCOTDataset("test")

    p_ref = float(train_ds.df[PRICE_COLS[0]].median())
    print(f"[Eval] Training p_ref = ${p_ref:.2f}/MWh")
    print(f"[Eval] TEST split     = {TEST_START} → {STAGE1_END}")
    print(f"[Eval] Val split ({VAL_START}→{TEST_START}) was used for")
    print(f"[Eval] checkpoint selection only — NOT reported here.\n")

    print("[1] Running training-median heuristic on TEST set...")
    heuristic = run_median_heuristic(test_ds, p_ref=p_ref, max_steps=18000)

    print("[2] Running PIO reference on TEST set...")
    pio = run_pio(test_ds, p_ref=p_ref, max_steps=2000)

    print("[3] Running SAC Plan C best checkpoint on TEST set...")
    sac = run_sac_agent(test_ds, p_ref=p_ref, max_steps=18000)

    results = [heuristic, pio, sac]
    print_results(results)

    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, "eval_results_planc_test.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved results → {out_path}")
    print("NOTE: These results use the TEST split — final reportable numbers.")


if __name__ == "__main__":
    main()
