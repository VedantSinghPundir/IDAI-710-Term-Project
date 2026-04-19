"""
Pipeline 5 — Evaluation & Baselines
======================================
Evaluates the trained Stage 1 agent against:
  1. Rule-based heuristic  (charge below median, discharge above)
  2. Perfect Information Optimisation (PIO) upper bound via LP
  3. Trained SAC agent

Prints a results table and saves results to logs/eval_results.csv.

Usage:
    python pipeline/p5_evaluate.py

Requirements:
    pip install scipy          (for LP baseline)
    checkpoints/stage1/stage1_best.pt  must exist (run p4_train.py first)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.config import *
from pipeline.p3_models import TTFE, Actor, Critic, FeasibilityProjection
from pipeline.p4_train  import ERCOTDataset, ERCOTEnv, SACAgent, flatten_obs


# ════════════════════════════════════════════════════════
# BASELINE 1 — Rule-Based Heuristic
# ════════════════════════════════════════════════════════

def run_rule_based(val_ds: ERCOTDataset) -> dict:
    """
    Simple price-threshold strategy:
      - Charge at full rate when RT LMP < median price
      - Discharge at full rate when RT LMP >= median price
    No look-ahead; uses current price only.
    """
    # Compute median from training period
    train_lmp = val_ds.df[PRICE_COLS[0]]
    median_price = float(train_lmp.median())

    env   = ERCOTEnv(val_ds)
    obs   = env.reset()
    pw, sv, tf, soc_arr = obs
    soc   = float(soc_arr[0])

    total_rev = 0.0
    revenues  = []
    n_steps   = 0

    while True:
        # Get current RT LMP (raw, not normalised)
        rt_lmp = val_ds.get_rt_lmp(env.idx)

        if rt_lmp < median_price:
            action = +1.0   # charge
        else:
            action = -1.0   # discharge

        # Apply SoC limits manually
        mwh_per_step = BATTERY_POWER_MW * INTERVAL_H
        delta_soc = action * mwh_per_step / BATTERY_CAP_MWH
        new_soc = np.clip(soc + delta_soc, SOC_MIN, SOC_MAX)
        actual_action = (new_soc - soc) * BATTERY_CAP_MWH / mwh_per_step

        (pw, sv, tf, soc_arr), reward, done = env.step(actual_action, new_soc)
        soc = float(soc_arr[0])
        total_rev += reward
        revenues.append(reward)
        n_steps += 1

        if done:
            break

    return {
        "name":         "Rule-based heuristic",
        "total_rev":    total_rev,
        "mean_rev":     np.mean(revenues),
        "std_rev":      np.std(revenues),
        "n_steps":      n_steps,
    }


# ════════════════════════════════════════════════════════
# BASELINE 2 — Perfect Information Optimisation (PIO)
# ════════════════════════════════════════════════════════

def run_pio(val_ds: ERCOTDataset, max_steps: int = 5000) -> dict:
    """
    Linear Program with perfect price foresight.
    This is the theoretical upper bound — the agent can never beat it.
    Uses scipy.optimize.linprog.

    Variables: charge_t, discharge_t ≥ 0 for each interval
    Objective: maximise Σ price_t × (discharge_t - charge_t) × INTERVAL_H
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        print("  [PIO] scipy not installed. Run: pip install scipy")
        return {"name": "PIO (skip)", "total_rev": float("nan"),
                "mean_rev": float("nan"), "std_rev": float("nan"), "n_steps": 0}

    T = min(max_steps, len(val_ds) - WINDOW_LEN - 1)
    prices = val_ds.df[PRICE_COLS[0]].iloc[WINDOW_LEN:WINDOW_LEN + T].values

    # Variables: [charge_0..T-1, discharge_0..T-1]
    # Maximise revenue = price × (discharge - charge) × INTERVAL_H
    # → minimise -price × (discharge - charge) × INTERVAL_H
    rev_per_mw = prices * INTERVAL_H
    c = np.concatenate([-rev_per_mw * (-1),   # charging costs (negative revenue)
                         -rev_per_mw * ( 1)])  # discharging earns

    # Bounds: 0 ≤ charge ≤ BATTERY_POWER_MW, 0 ≤ discharge ≤ BATTERY_POWER_MW
    bounds = [(0, BATTERY_POWER_MW)] * T + [(0, BATTERY_POWER_MW)] * T

    # SoC evolution: soc_t = soc_0 + Σ(charge - discharge) × INTERVAL_H / CAP
    # SoC constraints: SOC_MIN ≤ soc_t ≤ SOC_MAX for all t
    A_ub, b_ub_upper, b_ub_lower = [], [], []
    coeff = INTERVAL_H / BATTERY_CAP_MWH
    for t in range(T):
        row = np.zeros(2 * T)
        row[:t+1]   =  coeff    # charge cumsum
        row[T:T+t+1]= -coeff    # discharge cumsum
        A_ub.append(row)        # soc ≤ SOC_MAX
        A_ub.append(-row)       # -soc ≤ -SOC_MIN

    b_ub = np.array([SOC_MAX - 0.5] * T + [-(SOC_MIN - 0.5)] * T)

    result = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub,
                     bounds=bounds, method="highs")

    if result.success:
        charge    = result.x[:T]
        discharge = result.x[T:]
        revenues  = prices * (discharge - charge) * INTERVAL_H
        total_rev = revenues.sum()
    else:
        print(f"  [PIO] LP did not converge: {result.message}")
        total_rev = float("nan")
        revenues  = np.zeros(T)

    return {
        "name":      "PIO (perfect foresight LP)",
        "total_rev": float(total_rev),
        "mean_rev":  float(np.mean(revenues)),
        "std_rev":   float(np.std(revenues)),
        "n_steps":   T,
    }


# ════════════════════════════════════════════════════════
# TRAINED SAC AGENT EVALUATION
# ════════════════════════════════════════════════════════

def run_sac_agent(val_ds: ERCOTDataset, max_steps: int = 5000) -> dict:
    best_ckpt = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")
    if not os.path.exists(best_ckpt):
        # Fallback to most recent checkpoint
        ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
        if not ckpts:
            print("  [SAC] No checkpoint found. Run p4_train.py first.")
            return {"name": "SAC agent (no ckpt)", "total_rev": float("nan"),
                    "mean_rev": float("nan"), "std_rev": float("nan"), "n_steps": 0}
        best_ckpt = os.path.join(CHECKPOINT_DIR, ckpts[-1])

    agent = SACAgent()
    step  = agent.load(best_ckpt)
    print(f"  [SAC] Loaded checkpoint from step {step}")

    env   = ERCOTEnv(val_ds)
    obs   = env.reset()
    pw, sv, tf, soc_arr = obs
    soc_val = float(soc_arr[0])

    revenues = []
    n_steps  = 0

    while n_steps < max_steps:
        action, new_soc = agent.select_action(pw, sv, tf, soc_val, deterministic=True)
        (pw, sv, tf, soc_arr), reward, done = env.step(action, new_soc)
        soc_val = float(soc_arr[0])
        revenues.append(reward)
        n_steps += 1
        if done:
            break

    total_rev = float(np.sum(revenues))
    return {
        "name":      f"SAC agent (step {step})",
        "total_rev": total_rev,
        "mean_rev":  float(np.mean(revenues)),
        "std_rev":   float(np.std(revenues)),
        "n_steps":   n_steps,
    }


# ════════════════════════════════════════════════════════
# PRINT RESULTS TABLE
# ════════════════════════════════════════════════════════

def print_results(results: list[dict], pio_rev: float):
    print("\n" + "=" * 65)
    print("Evaluation Results")
    print("=" * 65)
    print(f"  {'Method':<35} {'Total Rev ($)':>14} {'vs Heuristic':>13} {'% of PIO':>10}")
    print(f"  {'-'*65}")

    heuristic_rev = next((r["total_rev"] for r in results
                          if "heuristic" in r["name"].lower()), 0)

    for r in results:
        rev = r["total_rev"]
        if np.isnan(rev):
            rev_str = "n/a"
            vs_str  = "n/a"
            pio_str = "n/a"
        else:
            rev_str = f"${rev:>12,.2f}"
            vs_str  = f"{(rev/heuristic_rev - 1)*100:>+.1f}%" if heuristic_rev else "n/a"
            pio_str = f"{rev/pio_rev*100:>8.1f}%"  if not np.isnan(pio_rev) and pio_rev else "n/a"
        print(f"  {r['name']:<35} {rev_str:>14} {vs_str:>13} {pio_str:>10}")

    print(f"\n  Note: PIO is the theoretical max (perfect foresight).")
    print(f"  SAC should exceed rule-based and approach PIO over training.")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Pipeline 5 — Evaluation")
    print("=" * 60)

    val_ds  = ERCOTDataset("val")
    MAX_STEPS = 5000

    print("\n[1] Running rule-based heuristic...")
    rule = run_rule_based(val_ds)
    print(f"    Total revenue: ${rule['total_rev']:,.2f}")

    print("\n[2] Running Perfect Information LP (PIO upper bound)...")
    pio  = run_pio(val_ds, max_steps=MAX_STEPS)
    print(f"    Total revenue: ${pio['total_rev']:,.2f}")

    print("\n[3] Running trained SAC agent (deterministic)...")
    sac  = run_sac_agent(val_ds, max_steps=MAX_STEPS)
    print(f"    Total revenue: ${sac['total_rev']:,.2f}")

    results = [rule, pio, sac]
    print_results(results, pio["total_rev"])

    # Save to CSV
    os.makedirs(LOG_DIR, exist_ok=True)
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(LOG_DIR, "eval_results.csv"), index=False)
    print(f"\n  Results saved → {LOG_DIR}/eval_results.csv")

    print("\n" + "=" * 60)
    print("✓ Evaluation complete.")
    print("Share these numbers with your teammates as your Stage 1 result.")


if __name__ == "__main__":
    main()
