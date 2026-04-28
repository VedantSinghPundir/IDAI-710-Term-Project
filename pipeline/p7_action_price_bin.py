"""
Pipeline 7 — Action-by-Price-Bin Analysis
==========================================
Proves the agent learned buy-low-sell-high behavior by showing
how actions change across price bins.

Price bins:
    Bin 1: price < $10/MWh          (very cheap — should charge aggressively)
    Bin 2: $10 – $20/MWh            (cheap)
    Bin 3: $20 – p_ref ($24.21)     (below median)
    Bin 4: p_ref – $40/MWh          (above median)
    Bin 5: $40 – $80/MWh            (expensive)
    Bin 6: $80 – $150/MWh           (very expensive)
    Bin 7: price > $150/MWh         (spike — discharge maximally)

For each bin reports:
    count       — how many val intervals fall in this bin
    mean_action — average action taken (-1=discharge, +1=charge)
    charge%     — fraction of intervals where action > 0.05
    discharge%  — fraction of intervals where action < -0.05
    hold%       — fraction of intervals where |action| <= 0.05
    mean_SoC    — average SoC when agent is in this bin
    mean_reward — average shaped reward received

Also compares agent vs heuristic per bin.

Usage:
    python pipeline/p7_action_price_bin.py
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.config import *
from pipeline.p3_models import FeasibilityProjection
from pipeline.p4_train  import ERCOTDataset, ERCOTEnv, SACAgent


# ════════════════════════════════════════════════════════
# PRICE BINS
# ════════════════════════════════════════════════════════

def make_bins(p_ref: float):
    """Returns list of (label, low, high) tuples."""
    return [
        (f"< $10",              -np.inf,  10.0),
        (f"$10 – $20",          10.0,     20.0),
        (f"$20 – ${p_ref:.0f} (p_ref)", 20.0, p_ref),
        (f"${p_ref:.0f} – $40", p_ref,   40.0),
        (f"$40 – $80",          40.0,    80.0),
        (f"$80 – $150",         80.0,   150.0),
        (f"> $150 (spike)",    150.0,  np.inf),
    ]


def assign_bin(price: float, bins: list) -> int:
    for i, (label, lo, hi) in enumerate(bins):
        if lo < price <= hi or (lo == -np.inf and price <= hi):
            return i
    return len(bins) - 1


# ════════════════════════════════════════════════════════
# ROLLOUT HELPERS
# ════════════════════════════════════════════════════════

def eval_step(env: ERCOTEnv, action: float, new_soc: float, p_ref: float):
    """Returns cash_reward, shaped_reward, degradation."""
    rt_lmp     = env.ds.get_rt_lmp(env.idx)
    grid_mwh   = -action * BATTERY_POWER_MW * INTERVAL_H
    cash       = grid_mwh * rt_lmp
    shaped     = grid_mwh * (rt_lmp - p_ref) / REWARD_SCALE - \
                 CYCLE_COST_PER_MWH * abs(grid_mwh) / REWARD_SCALE
    degradation= CYCLE_COST_PER_MWH * abs(grid_mwh)

    env.soc       = float(np.clip(new_soc, SOC_MIN, SOC_MAX))
    env.idx      += 1
    env.ep_steps += 1
    done = (env.idx >= len(env.ds) - 1)
    return env._obs(), cash, shaped, done, degradation


def rollout(env: ERCOTEnv, action_fn, max_steps: int, p_ref: float):
    """
    Generic rollout. action_fn(env, obs) → (action, new_soc).
    Returns list of records per step.
    """
    obs = env.reset_deterministic()
    pw, sv, tf, soc_arr = obs
    soc_val = float(soc_arr[0])

    records = []
    for _ in range(max_steps):
        rt_lmp = env.ds.get_rt_lmp(env.idx)
        action, new_soc = action_fn(env, pw, sv, tf, soc_val)

        (pw, sv, tf, soc_arr), cash, shaped, done, deg = eval_step(
            env, action, new_soc, p_ref
        )

        records.append({
            "price":       rt_lmp,
            "action":      action,
            "soc_before":  soc_val,
            "cash":        cash,
            "shaped":      shaped,
            "degradation": deg,
        })

        soc_val = float(soc_arr[0])
        if done:
            break

    return records


# ════════════════════════════════════════════════════════
# BIN ANALYSIS
# ════════════════════════════════════════════════════════

HOLD_THRESH = 0.05   # |action| < this → classified as hold


def analyse_bins(records: list, bins: list, name: str) -> pd.DataFrame:
    rows = []
    for i, (label, lo, hi) in enumerate(bins):
        bin_recs = [r for r in records
                    if (r["price"] > lo or lo == -np.inf) and r["price"] <= hi]
        if not bin_recs:
            continue

        actions    = np.array([r["action"]     for r in bin_recs])
        socs       = np.array([r["soc_before"] for r in bin_recs])
        cash_arr   = np.array([r["cash"]       for r in bin_recs])
        shaped_arr = np.array([r["shaped"]     for r in bin_recs])

        charge_pct    = (actions >  HOLD_THRESH).mean() * 100
        discharge_pct = (actions < -HOLD_THRESH).mean() * 100
        hold_pct      = 100 - charge_pct - discharge_pct

        rows.append({
            "bin":         label,
            "count":       len(bin_recs),
            "mean_price":  np.mean([r["price"] for r in bin_recs]),
            "mean_action": actions.mean(),
            "charge%":     charge_pct,
            "discharge%":  discharge_pct,
            "hold%":       hold_pct,
            "mean_SoC":    socs.mean(),
            "mean_cash":   cash_arr.mean(),
            "sum_cash":    cash_arr.sum(),
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("Pipeline 7 — Action-by-Price-Bin Analysis")
    print("=" * 75)

    # ── Load ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    agent = SACAgent()
    step  = agent.load(ckpt_path)
    print(f"  Checkpoint : step {step:,}")

    val_ds   = ERCOTDataset("val")
    train_ds = ERCOTDataset("train")
    p_ref    = float(train_ds.df[PRICE_COLS[0]].median())
    print(f"  p_ref      : ${p_ref:.2f}/MWh")
    print(f"  Val rows   : {len(val_ds):,}")

    bins = make_bins(p_ref)
    proj = FeasibilityProjection().to(DEVICE)

    # ── SAC rollout ────────────────────────────────────────────────
    def sac_action(env, pw, sv, tf, soc_val):
        action, new_soc = agent.select_action(pw, sv, tf, soc_val, deterministic=True)
        return action, new_soc

    sac_env = ERCOTEnv(val_ds, p_ref=p_ref)
    print("\n[1] Running SAC agent rollout (18,000 steps)...")
    sac_records = rollout(sac_env, sac_action, max_steps=18000, p_ref=p_ref)

    # ── Heuristic rollout ──────────────────────────────────────────
    def heuristic_action(env, pw, sv, tf, soc_val):
        rt_lmp   = env.ds.get_rt_lmp(env.idx)
        raw      = 1.0 if rt_lmp < p_ref else -1.0
        a_t      = torch.FloatTensor([[raw]]).to(DEVICE)
        s_t      = torch.FloatTensor([[soc_val]]).to(DEVICE)
        with torch.no_grad():
            f_t, ns_t = proj(a_t, s_t)
        return f_t.item(), ns_t.item()

    heur_env = ERCOTEnv(val_ds, p_ref=p_ref)
    print("[2] Running heuristic rollout (18,000 steps)...")
    heur_records = rollout(heur_env, heuristic_action, max_steps=18000, p_ref=p_ref)

    # ── Bin analysis ───────────────────────────────────────────────
    sac_df  = analyse_bins(sac_records,  bins, "SAC")
    heur_df = analyse_bins(heur_records, bins, "Heuristic")

    # ── Print results ──────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SAC AGENT — Actions by Price Bin")
    print("=" * 75)
    print(f"  {'Price Bin':<26} {'N':>5} {'MeanAct':>9} {'Charge%':>9} "
          f"{'Dis%':>7} {'Hold%':>7} {'MeanSoC':>8} {'$/step':>9}")
    print(f"  {'-'*75}")
    for _, row in sac_df.iterrows():
        expected = ("CHG↑" if row["mean_price"] < p_ref else "DIS↓")
        direction_ok = (
            (row["mean_price"] < p_ref  and row["mean_action"] > 0) or
            (row["mean_price"] >= p_ref and row["mean_action"] < 0)
        )
        ok = "✓" if direction_ok else "✗"
        print(f"  {row['bin']:<26} {row['count']:>5} {row['mean_action']:>+9.3f} "
              f"{row['charge%']:>8.1f}% {row['discharge%']:>6.1f}% "
              f"{row['hold%']:>6.1f}% {row['mean_SoC']:>8.3f} "
              f"{row['mean_cash']:>8.3f} {ok} ({expected})")

    print("\n" + "=" * 75)
    print("HEURISTIC — Actions by Price Bin")
    print("=" * 75)
    print(f"  {'Price Bin':<26} {'N':>5} {'MeanAct':>9} {'Charge%':>9} "
          f"{'Dis%':>7} {'Hold%':>7} {'MeanSoC':>8} {'$/step':>9}")
    print(f"  {'-'*75}")
    for _, row in heur_df.iterrows():
        expected = ("CHG↑" if row["mean_price"] < p_ref else "DIS↓")
        direction_ok = (
            (row["mean_price"] < p_ref  and row["mean_action"] > 0) or
            (row["mean_price"] >= p_ref and row["mean_action"] < 0)
        )
        ok = "✓" if direction_ok else "✗"
        print(f"  {row['bin']:<26} {row['count']:>5} {row['mean_action']:>+9.3f} "
              f"{row['charge%']:>8.1f}% {row['discharge%']:>6.1f}% "
              f"{row['hold%']:>6.1f}% {row['mean_SoC']:>8.3f} "
              f"{row['mean_cash']:>8.3f} {ok} ({expected})")

    # ── Comparison table ───────────────────────────────────────────
    print("\n" + "=" * 75)
    print("COMPARISON — SAC vs Heuristic Mean Action per Bin")
    print("=" * 75)
    print(f"  {'Price Bin':<26} {'SAC action':>12} {'Heur action':>12} "
          f"{'SAC chg%':>10} {'Heur chg%':>10}")
    print(f"  {'-'*75}")
    for i, (label, lo, hi) in enumerate(bins):
        sac_row  = sac_df[sac_df["bin"]  == label]
        heur_row = heur_df[heur_df["bin"] == label]
        if sac_row.empty or heur_row.empty:
            continue
        print(f"  {label:<26} "
              f"{sac_row['mean_action'].values[0]:>+12.3f} "
              f"{heur_row['mean_action'].values[0]:>+12.3f} "
              f"{sac_row['charge%'].values[0]:>9.1f}% "
              f"{heur_row['charge%'].values[0]:>9.1f}%")

    # ── Key insight summary ────────────────────────────────────────
    print("\n" + "=" * 75)
    print("KEY BEHAVIORAL INSIGHTS")
    print("=" * 75)

    # Check if SAC charges more at low prices than heuristic
    low_bins  = [b[0] for b in bins if b[2] <= p_ref]
    high_bins = [b[0] for b in bins if b[1] >= p_ref]

    sac_low_chg  = sac_df[sac_df["bin"].isin(low_bins)]["charge%"].mean()
    heur_low_chg = heur_df[heur_df["bin"].isin(low_bins)]["charge%"].mean()
    sac_hi_dis   = sac_df[sac_df["bin"].isin(high_bins)]["discharge%"].mean()
    heur_hi_dis  = heur_df[heur_df["bin"].isin(high_bins)]["discharge%"].mean()

    print(f"  Avg charge%    at LOW prices  (<p_ref):  "
          f"SAC={sac_low_chg:.1f}%  Heuristic={heur_low_chg:.1f}%")
    print(f"  Avg discharge% at HIGH prices (≥p_ref):  "
          f"SAC={sac_hi_dis:.1f}%  Heuristic={heur_hi_dis:.1f}%")

    if sac_low_chg > heur_low_chg:
        print(f"\n  ✓ SAC charges MORE at low prices than heuristic")
    else:
        print(f"\n  ✗ SAC charges LESS at low prices than heuristic")

    if sac_hi_dis > heur_hi_dis:
        print(f"  ✓ SAC discharges MORE at high prices than heuristic")
    else:
        print(f"  ✗ SAC discharges LESS at high prices than heuristic")

    # ── Save CSV ───────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    sac_df["method"]  = "SAC"
    heur_df["method"] = "Heuristic"
    combined = pd.concat([sac_df, heur_df], ignore_index=True)
    out_path = os.path.join(LOG_DIR, "action_price_bin.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")

    print("\n" + "=" * 75)
    print("✓ Action-by-price-bin analysis complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()
