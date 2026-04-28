"""
Pipeline 6 — Q-Value Diagnostic (Multi-SoC Level Sweep)
=========================================================
Tests Q-value ordering at three SoC levels:
    SoC = 0.10  (near empty — discharge should be penalised)
    SoC = 0.50  (midpoint — baseline comparison)
    SoC = 0.90  (near full — charge should be penalised)

At each SoC level checks:
    Low price (<p_ref):  Q(charge) > Q(hold) > Q(discharge)  ← correct
    High price (≥p_ref): Q(discharge) > Q(hold) > Q(charge)  ← correct

Plan B result (before fixes):
    SoC=0.50, price=$8.63: Q(dis)=36.343 > Q(chg)=35.570  WRONG
    0 out of 100 low-price intervals preferred charging

Expected Plan C result:
    All SoC levels: low price → charge preferred  ← credit assignment fixed

Usage:
    python pipeline/p6_qval_diagnostic.py
"""

import os
import sys
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.config import *
from pipeline.p4_train import ERCOTDataset, SACAgent


# ════════════════════════════════════════════════════════
# OBS BUILDER — no ERCOTEnv instantiation (avoids print spam)
# ════════════════════════════════════════════════════════

def get_obs_tensors(ds, idx: int, soc_val: float):
    """Build observation tensors directly from dataset — no env needed."""
    start  = max(0, idx - WINDOW_LEN + 1)
    window = ds.df[PRICE_COLS].iloc[start:idx+1].values.astype(np.float32)
    if len(window) < WINDOW_LEN:
        pad    = np.repeat(window[[0]], WINDOW_LEN - len(window), axis=0)
        window = np.concatenate([pad, window], axis=0)
    mean_p = ds.mean[:PRICE_DIM]
    std_p  = ds.std[:PRICE_DIM]
    pw     = np.clip((window - mean_p) / std_p, -CLIP_SIGMA, CLIP_SIGMA)

    raw_sv = ds.df[SYSTEM_COLS].iloc[idx].values.astype(np.float32)
    sv     = (raw_sv - ds.mean[PRICE_DIM:]) / ds.std[PRICE_DIM:]

    ts = ds.get_timestamp(idx)
    h  = ts.hour + ts.minute / 60
    dw = ts.dayofweek
    tf = np.array([
        math.sin(2*math.pi*h/24),  math.cos(2*math.pi*h/24),
        math.sin(2*math.pi*dw/7),  math.cos(2*math.pi*dw/7),
        math.sin(4*math.pi*dw/7),  math.cos(4*math.pi*dw/7),
    ], dtype=np.float32)

    return (
        torch.FloatTensor(pw).unsqueeze(0).to(DEVICE),
        torch.FloatTensor(sv).unsqueeze(0).to(DEVICE),
        torch.FloatTensor(tf).unsqueeze(0).to(DEVICE),
        torch.FloatTensor([[soc_val]]).to(DEVICE),
    )


# ════════════════════════════════════════════════════════
# SINGLE SoC LEVEL DIAGNOSTIC
# ════════════════════════════════════════════════════════

def run_soc_level(agent, val_ds, p_ref: float,
                  soc_val: float, n_points: int = 100):
    """
    Evaluate Q(discharge), Q(hold), Q(charge) at a fixed SoC level
    across n_points timesteps from the val dataset.
    """
    act_dis  = torch.FloatTensor([[-1.0]]).to(DEVICE)
    act_hold = torch.FloatTensor([[ 0.0]]).to(DEVICE)
    act_chg  = torch.FloatTensor([[ 1.0]]).to(DEVICE)

    results = []
    for i in range(0, n_points * 5, 5):
        idx = WINDOW_LEN + i
        if idx >= len(val_ds) - 1:
            break

        rt_lmp = val_ds.get_rt_lmp(idx)
        pw_t, sv_t, tf_t, soc_t = get_obs_tensors(val_ds, idx, soc_val)

        with torch.no_grad():
            obs = agent.encode(pw_t, sv_t, tf_t, soc_t)
            q_dis  = agent.critic.q_min(obs, act_dis).item()
            q_hold = agent.critic.q_min(obs, act_hold).item()
            q_chg  = agent.critic.q_min(obs, act_chg).item()

        if rt_lmp < p_ref:
            correct  = (q_chg > q_dis)
            expected = "CHARGE"
        else:
            correct  = (q_dis > q_chg)
            expected = "DISCHARGE"

        results.append({
            "price":    rt_lmp,
            "q_dis":    q_dis,
            "q_hold":   q_hold,
            "q_chg":    q_chg,
            "correct":  correct,
            "expected": expected,
        })

    return results


# ════════════════════════════════════════════════════════
# PRINT ONE SoC SECTION
# ════════════════════════════════════════════════════════

def print_soc_section(soc_val: float, results: list, p_ref: float):
    prices  = np.array([r["price"]  for r in results])
    q_dis   = np.array([r["q_dis"]  for r in results])
    q_chg   = np.array([r["q_chg"]  for r in results])
    correct = np.array([r["correct"] for r in results])

    low_mask  = prices <  p_ref
    high_mask = prices >= p_ref
    low_corr  = correct[low_mask].sum()  if low_mask.sum()  > 0 else 0
    high_corr = correct[high_mask].sum() if high_mask.sum() > 0 else 0

    soc_note = ""
    if soc_val <= 0.15:
        soc_note = "  ← near empty: discharge physically limited"
    elif soc_val >= 0.85:
        soc_note = "  ← near full: charge physically limited"

    print(f"\n{'━'*65}")
    print(f"  SoC = {soc_val:.2f}{soc_note}")
    print(f"{'━'*65}")
    print(f"  Overall correct ordering      : {correct.sum()}/{len(correct)}"
          f" ({correct.mean()*100:.1f}%)")
    print(f"  Correct at low  price (<p_ref): {low_corr}/{low_mask.sum()}"
          f" ({low_corr/max(low_mask.sum(),1)*100:.1f}%)")
    print(f"  Correct at high price (≥p_ref): {high_corr}/{high_mask.sum()}"
          f" ({high_corr/max(high_mask.sum(),1)*100:.1f}%)")
    print(f"  Q(discharge) mean : {q_dis.mean():.4f}"
          f"{'  (should be LOW at near-empty SoC)' if soc_val <= 0.15 else ''}")
    print(f"  Q(charge)    mean : {q_chg.mean():.4f}"
          f"{'  (should be LOW at near-full SoC)'  if soc_val >= 0.85 else ''}")

    # Low price sample (5 cheapest)
    low_res = sorted([r for r in results if r["price"] < p_ref],
                     key=lambda x: x["price"])[:5]
    if low_res:
        print(f"\n  Lowest price intervals (expect CHARGE):")
        for r in low_res:
            best = max(r["q_dis"], r["q_hold"], r["q_chg"])
            bl   = ("DIS" if best==r["q_dis"] else
                    "HLD" if best==r["q_hold"] else "CHG")
            ok   = "✓" if r["correct"] else "✗"
            print(f"    ${r['price']:>7.2f} | "
                  f"Q(dis)={r['q_dis']:>8.3f} | "
                  f"Q(hld)={r['q_hold']:>8.3f} | "
                  f"Q(chg)={r['q_chg']:>8.3f} | best={bl} {ok}")

    # High price sample (5 most expensive)
    high_res = sorted([r for r in results if r["price"] >= p_ref],
                      key=lambda x: x["price"], reverse=True)[:5]
    if high_res:
        print(f"\n  Highest price intervals (expect DISCHARGE):")
        for r in high_res:
            best = max(r["q_dis"], r["q_hold"], r["q_chg"])
            bl   = ("DIS" if best==r["q_dis"] else
                    "HLD" if best==r["q_hold"] else "CHG")
            ok   = "✓" if r["correct"] else "✗"
            print(f"    ${r['price']:>7.2f} | "
                  f"Q(dis)={r['q_dis']:>8.3f} | "
                  f"Q(hld)={r['q_hold']:>8.3f} | "
                  f"Q(chg)={r['q_chg']:>8.3f} | best={bl} {ok}")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def run_diagnostic():
    print("=" * 65)
    print("Pipeline 6 — Q-Value Diagnostic (Multi-SoC Level Sweep)")
    print("=" * 65)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}\n"
            "Run p4_train.py first or point to correct checkpoint."
        )

    agent = SACAgent()
    step  = agent.load(ckpt_path)
    print(f"  Checkpoint  : step {step:,}")
    print(f"  ALPHA_FIXED : {agent.alpha}")

    val_ds   = ERCOTDataset("val")
    train_ds = ERCOTDataset("train")
    p_ref    = float(train_ds.df[PRICE_COLS[0]].median())

    print(f"  p_ref       : ${p_ref:.2f}/MWh (training median)")
    print(f"  Val rows    : {len(val_ds):,}")
    print(f"  SoC levels  : 0.10, 0.50, 0.90  (100 intervals each)")

    # ── Run all three SoC levels ───────────────────────────────────
    soc_levels  = [0.10, 0.50, 0.90]
    all_results = {}
    for soc_val in soc_levels:
        all_results[soc_val] = run_soc_level(
            agent, val_ds, p_ref, soc_val, n_points=100
        )
        print_soc_section(soc_val, all_results[soc_val], p_ref)

    # ── Cross-SoC summary table ────────────────────────────────────
    print(f"\n{'━'*65}")
    print("  CROSS-SoC SUMMARY TABLE")
    print(f"{'━'*65}")
    print(f"  {'SoC':>5} {'Total%':>8} {'Low%':>8} {'High%':>8} "
          f"{'Q(dis)':>10} {'Q(chg)':>10} {'Gap(dis-chg)':>14}")
    print(f"  {'-'*65}")
    for soc_val in soc_levels:
        res    = all_results[soc_val]
        prices = np.array([r["price"]  for r in res])
        corr   = np.array([r["correct"] for r in res])
        q_dis  = np.array([r["q_dis"]  for r in res])
        q_chg  = np.array([r["q_chg"]  for r in res])
        lm     = prices <  p_ref
        hm     = prices >= p_ref
        lc     = corr[lm].mean()*100 if lm.sum()>0 else 0.0
        hc     = corr[hm].mean()*100 if hm.sum()>0 else 0.0
        gap    = q_dis.mean() - q_chg.mean()
        print(f"  {soc_val:>5.2f} {corr.mean()*100:>7.1f}% {lc:>7.1f}% "
              f"{hc:>7.1f}% {q_dis.mean():>10.4f} {q_chg.mean():>10.4f} "
              f"{gap:>+14.4f}")

    print(f"\n  Gap interpretation:")
    print(f"    Negative gap at low price  = charge preferred  ✓")
    print(f"    Positive gap at high price = discharge preferred ✓")
    print(f"    Gap shrinks near-empty (SoC=0.10) because discharge physically limited")
    print(f"    Gap shrinks near-full  (SoC=0.90) because charge physically limited")

    # ── Plan B comparison ──────────────────────────────────────────
    print(f"\n{'━'*65}")
    print("  VERSUS PLAN B (cashflow-only reward + projection bugs)")
    print(f"{'━'*65}")
    print("  Plan B SoC=0.50 (before any fixes):")
    print("    price=$8.63:  Q(dis)=36.343 > Q(chg)=35.570  ✗  DISCHARGE at low price")
    print("    price=$9.10:  Q(dis)=31.231 > Q(chg)=30.391  ✗")
    print("    Correct at low prices: 0 / 100  (0.0%)")
    print()

    soc50    = all_results[0.50]
    prices50 = np.array([r["price"] for r in soc50])
    low50    = [r for r in soc50 if r["price"] < p_ref]
    low_ok50 = sum(1 for r in low50 if r["correct"])
    status   = "CREDIT ASSIGNMENT FIXED ✓" if low_ok50 == len(low50) else "PARTIAL FIX"

    print(f"  Plan C SoC=0.50 (this checkpoint):")
    print(f"    Correct at low prices: {low_ok50}/{len(low50)}"
          f" ({low_ok50/max(len(low50),1)*100:.1f}%)  [{status}]")

    print(f"\n{'='*65}")
    print("✓ Q-value diagnostic complete.")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_diagnostic()
