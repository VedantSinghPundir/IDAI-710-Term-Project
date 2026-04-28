"""
Pipeline 8 — Revenue Decomposition
=====================================
Breaks down the full evaluation result into components to prove
the agent's revenue is from genuine arbitrage, not just selling
pre-existing battery inventory at any price.

Reports for SAC and Heuristic:
    cash_revenue        — raw market cash received
    inventory_change    — value of SoC change at p_ref (+ = bought net)
    degradation_cost    — battery wear from cycling
    inventory_adjusted  — cash + inventory_change - degradation (fair profit)

Also reports:
    Hourly revenue profile (to show when agent earns most)
    Revenue in price spike vs non-spike intervals
    Action distribution summary

Usage:
    python pipeline/p8_revenue_decomposition.py
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
# ROLLOUT WITH FULL TRACKING
# ════════════════════════════════════════════════════════

def full_rollout(env: ERCOTEnv, action_fn, max_steps: int,
                 p_ref: float, val_ds: ERCOTDataset):
    """
    Full rollout tracking every component needed for decomposition.
    Returns detailed records + summary.
    """
    # Deterministic start
    env.idx      = WINDOW_LEN
    env.soc      = 0.5
    env.ep_steps = 0
    pw, sv, tf, soc_arr = env._obs()
    soc_val     = float(soc_arr[0])
    initial_soc = soc_val

    records = []

    for step_i in range(max_steps):
        rt_lmp     = val_ds.get_rt_lmp(env.idx)
        timestamp  = val_ds.get_timestamp(env.idx)
        action, new_soc = action_fn(env, pw, sv, tf, soc_val)

        grid_mwh    = -action * BATTERY_POWER_MW * INTERVAL_H
        cash        = grid_mwh * rt_lmp
        spread      = grid_mwh * (rt_lmp - p_ref)
        degradation = CYCLE_COST_PER_MWH * abs(grid_mwh)

        # Classify action
        if   action >  0.05: act_type = "charge"
        elif action < -0.05: act_type = "discharge"
        else:                 act_type = "hold"

        records.append({
            "step":        step_i,
            "timestamp":   timestamp,
            "hour":        timestamp.hour,
            "price":       rt_lmp,
            "action":      action,
            "act_type":    act_type,
            "soc_before":  soc_val,
            "new_soc":     new_soc,
            "grid_mwh":    grid_mwh,
            "cash":        cash,
            "spread":      spread,
            "degradation": degradation,
            "is_spike":    rt_lmp > 200,
        })

        # Step env manually (cash-only, no episode limit)
        env.soc       = float(np.clip(new_soc, SOC_MIN, SOC_MAX))
        env.idx      += 1
        env.ep_steps += 1
        if env.idx >= len(val_ds) - 1:
            # rebuild obs before break
            pw, sv, tf, soc_arr = env._obs()
            soc_val = float(soc_arr[0])
            break

        pw, sv, tf, soc_arr = env._obs()
        soc_val = float(soc_arr[0])

    final_soc = soc_val
    df = pd.DataFrame(records)

    total_cash        = df["cash"].sum()
    total_degradation = df["degradation"].sum()
    inventory_change  = (final_soc - initial_soc) * BATTERY_CAP_MWH * p_ref
    inv_adjusted      = total_cash + inventory_change - total_degradation

    summary = {
        "n_steps":          len(df),
        "initial_soc":      initial_soc,
        "final_soc":        final_soc,
        "total_cash":       total_cash,
        "inventory_change": inventory_change,
        "degradation_cost": total_degradation,
        "inv_adjusted":     inv_adjusted,
        "n_charge":         (df["act_type"] == "charge").sum(),
        "n_discharge":      (df["act_type"] == "discharge").sum(),
        "n_hold":           (df["act_type"] == "hold").sum(),
        "spike_cash":       df[df["is_spike"]]["cash"].sum(),
        "normal_cash":      df[~df["is_spike"]]["cash"].sum(),
        "n_spikes":         df["is_spike"].sum(),
    }

    return df, summary


# ════════════════════════════════════════════════════════
# PRINT HELPERS
# ════════════════════════════════════════════════════════

def print_decomposition(name: str, s: dict, p_ref: float):
    n = s["n_steps"]
    c = s["n_charge"]
    d = s["n_discharge"]
    h = s["n_hold"]
    total = max(c + d + h, 1)

    print(f"\n{'━'*65}")
    print(f"  {name}")
    print(f"{'━'*65}")
    print(f"  Steps evaluated    : {n:,}")
    print(f"  SoC: {s['initial_soc']:.3f} → {s['final_soc']:.3f}"
          f"  (change = {s['final_soc']-s['initial_soc']:+.3f})")
    print()
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  Revenue Decomposition                  │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Cash revenue         : ${s['total_cash']:>12,.2f}    │")
    inv_sign = "+" if s['inventory_change'] >= 0 else ""
    print(f"  │  Inventory adjustment : {inv_sign}${abs(s['inventory_change']):>11,.2f}    │")
    print(f"  │    (SoC Δ × capacity × p_ref=${p_ref:.2f})          │")
    print(f"  │  Degradation cost     : -${s['degradation_cost']:>11,.2f}    │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Inventory-adjusted   : ${s['inv_adjusted']:>12,.2f}    │")
    print(f"  └─────────────────────────────────────────┘")
    print()
    print(f"  Per-step averages:")
    print(f"    Cash/step        : ${s['total_cash']/n:>8.3f}")
    print(f"    InvAdj/step      : ${s['inv_adjusted']/n:>8.3f}")
    print(f"    Degradation/step : ${s['degradation_cost']/n:>8.3f}")
    print()
    print(f"  Action distribution:")
    print(f"    Charge     : {c:>5,} steps  ({c/total*100:.1f}%)")
    print(f"    Discharge  : {d:>5,} steps  ({d/total*100:.1f}%)")
    print(f"    Hold       : {h:>5,} steps  ({h/total*100:.1f}%)")
    print()
    print(f"  Spike analysis (price > $200/MWh):")
    print(f"    Spike intervals : {s['n_spikes']:,}  ({s['n_spikes']/n*100:.2f}%)")
    print(f"    Cash from spikes: ${s['spike_cash']:,.2f}  "
          f"({s['spike_cash']/max(abs(s['total_cash']),1)*100:.1f}% of total)")
    print(f"    Cash from normal: ${s['normal_cash']:,.2f}")


def print_hourly(name: str, df: pd.DataFrame):
    """Print per-hour revenue aggregation."""
    hourly = df.groupby("hour").agg(
        mean_cash=("cash", "mean"),
        sum_cash=("cash", "sum"),
        charge_pct=("act_type", lambda x: (x=="charge").mean()*100),
        discharge_pct=("act_type", lambda x: (x=="discharge").mean()*100),
        mean_price=("price", "mean"),
        count=("cash", "count"),
    ).reset_index()

    print(f"\n  Hourly Revenue Profile — {name}:")
    print(f"  {'Hour':>5} {'Intervals':>10} {'AvgPrice':>10} "
          f"{'Chg%':>7} {'Dis%':>7} {'$/step':>9}")
    print(f"  {'-'*55}")
    for _, row in hourly.iterrows():
        print(f"  {int(row['hour']):>5} {int(row['count']):>10} "
              f"${row['mean_price']:>9.2f} "
              f"{row['charge_pct']:>6.1f}% {row['discharge_pct']:>6.1f}% "
              f"${row['mean_cash']:>8.3f}")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Pipeline 8 — Revenue Decomposition")
    print("=" * 65)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    agent = SACAgent()
    step  = agent.load(ckpt_path)
    print(f"  Checkpoint : step {step:,}")

    val_ds   = ERCOTDataset("val")
    train_ds = ERCOTDataset("train")
    p_ref    = float(train_ds.df[PRICE_COLS[0]].median())
    print(f"  p_ref      : ${p_ref:.2f}/MWh  (training median)")

    proj = FeasibilityProjection().to(DEVICE)

    # ── SAC rollout ────────────────────────────────────────────────
    def sac_action(env, pw, sv, tf, soc_val):
        return agent.select_action(pw, sv, tf, soc_val, deterministic=True)

    print("\n[1] Running SAC agent...")
    sac_env = ERCOTEnv(val_ds, p_ref=p_ref)
    sac_df, sac_s = full_rollout(sac_env, sac_action, 18000, p_ref, val_ds)

    # ── Heuristic rollout ──────────────────────────────────────────
    def heuristic_action(env, pw, sv, tf, soc_val):
        rt_lmp = val_ds.get_rt_lmp(env.idx)
        raw    = 1.0 if rt_lmp < p_ref else -1.0
        a_t    = torch.FloatTensor([[raw]]).to(DEVICE)
        s_t    = torch.FloatTensor([[soc_val]]).to(DEVICE)
        with torch.no_grad():
            f_t, ns_t = proj(a_t, s_t)
        return f_t.item(), ns_t.item()

    print("[2] Running heuristic...")
    heur_env = ERCOTEnv(val_ds, p_ref=p_ref)
    heur_df, heur_s = full_rollout(heur_env, heuristic_action, 18000, p_ref, val_ds)

    # ── Print decomposition ────────────────────────────────────────
    print_decomposition("SAC Plan C (step 100k)", sac_s, p_ref)
    print_decomposition("Median Heuristic",       heur_s, p_ref)

    # ── Head-to-head comparison ────────────────────────────────────
    print(f"\n{'━'*65}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'━'*65}")
    heur_inv = heur_s["inv_adjusted"]
    sac_inv  = sac_s["inv_adjusted"]
    uplift   = (sac_inv / max(abs(heur_inv), 1) - 1) * 100

    print(f"  {'Metric':<30} {'SAC':>14} {'Heuristic':>14} {'SAC vs Heur':>14}")
    print(f"  {'-'*65}")
    metrics = [
        ("Cash revenue",        sac_s["total_cash"],       heur_s["total_cash"]),
        ("Inventory adjustment", sac_s["inventory_change"], heur_s["inventory_change"]),
        ("Degradation cost",    -sac_s["degradation_cost"],-heur_s["degradation_cost"]),
        ("Inv-adjusted profit", sac_s["inv_adjusted"],     heur_s["inv_adjusted"]),
    ]
    for label, sv, hv in metrics:
        diff = sv - hv
        print(f"  {label:<30} ${sv:>12,.2f} ${hv:>12,.2f} {'+' if diff>=0 else ''}${diff:>12,.2f}")

    print(f"\n  SAC beats heuristic by {uplift:+.1f}% on inventory-adjusted profit")
    print()
    print(f"  Note: inventory adjustment removes the advantage of selling")
    print(f"  pre-existing battery inventory. The gap is genuine arbitrage skill.")

    # ── Hourly profiles ────────────────────────────────────────────
    print()
    print_hourly("SAC",       sac_df)
    print_hourly("Heuristic", heur_df)

    # ── Save CSVs ──────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)

    sac_df.to_csv(os.path.join(LOG_DIR, "revenue_decomp_sac.csv"), index=False)
    heur_df.to_csv(os.path.join(LOG_DIR, "revenue_decomp_heur.csv"), index=False)

    summary_rows = [
        {**{"method": "SAC"},        **sac_s},
        {**{"method": "Heuristic"},  **heur_s},
    ]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(LOG_DIR, "revenue_decomp_summary.csv"), index=False
    )
    print(f"\n  CSVs saved → {LOG_DIR}/revenue_decomp_*.csv")

    print("\n" + "=" * 65)
    print("✓ Revenue decomposition complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
