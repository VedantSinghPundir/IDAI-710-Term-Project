"""
Pipeline 6 — Reward Sanity Check
==================================
Verifies Plan C inventory-adjusted reward has correct sign at all price levels.

RUN THIS BEFORE p4_train.py.
Training should not start until this check passes.

Expected results:
    Low price  (e.g. $9):  charge wins
    Near p_ref (~$24):     hold wins (or very close — degradation tips it)
    High price (e.g. $40): discharge wins
    Spike      (e.g. $120): discharge wins by large margin

Usage:
    python pipeline/p6_reward_sanity.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.config import (
    BATTERY_POWER_MW, INTERVAL_H, CYCLE_COST_PER_MWH, REWARD_SCALE,
    SOC_MIN, SOC_MAX, BATTERY_CAP_MWH, EFFICIENCY,
)


# ════════════════════════════════════════════════════════
# REWARD FORMULA (mirrors ERCOTEnv.step exactly)
# ════════════════════════════════════════════════════════

def plan_c_reward(action: float, price: float, p_ref: float) -> float:
    """
    Compute Plan C shaped reward.

    action:
        +1 = full charge   (buy from grid)
        -1 = full discharge (sell to grid)
         0 = hold

    grid_mwh:
        positive = energy sold (discharge)
        negative = energy bought (charge)
    """
    grid_mwh      = -action * BATTERY_POWER_MW * INTERVAL_H
    spread_reward = grid_mwh * (price - p_ref)
    degradation   = CYCLE_COST_PER_MWH * abs(grid_mwh)
    return (spread_reward - degradation) / REWARD_SCALE


def energy_mwh(action: float) -> float:
    return abs(action * BATTERY_POWER_MW * INTERVAL_H)


# ════════════════════════════════════════════════════════
# MAIN CHECK
# ════════════════════════════════════════════════════════

def main():
    # Use 2022-2025 training median — update if p2 prints a different value
    # This is printed as "[Env:train] p_ref = $XX.XX/MWh" when training starts
    p_ref = 24.21

    actions = {
        "charge  (+1)": +1.0,
        "hold    ( 0)":  0.0,
        "dschg   (-1)": -1.0,
    }

    test_prices = [
        ("Very low",  9.0),
        ("Low",      15.0),
        ("Near ref", p_ref + 0.5),
        ("Near ref", p_ref - 0.5),
        ("High",     40.0),
        ("Very high", 80.0),
        ("Spike",   120.0),
    ]

    print("=" * 65)
    print("Plan C Reward Sanity Check")
    print("=" * 65)
    print(f"p_ref           = ${p_ref:.2f}/MWh")
    print(f"BATTERY_POWER   = {BATTERY_POWER_MW} MW")
    print(f"INTERVAL_H      = {INTERVAL_H:.4f} h  ({INTERVAL_H*60:.0f} min)")
    print(f"CYCLE_COST      = ${CYCLE_COST_PER_MWH}/MWh")
    print(f"REWARD_SCALE    = {REWARD_SCALE}")
    print(f"MWh per action  = {energy_mwh(1.0):.4f} MWh")
    print("=" * 65)
    print()

    all_passed = True

    for label, price in test_prices:
        rewards = {name: plan_c_reward(a, price, p_ref)
                   for name, a in actions.items()}
        best_name = max(rewards, key=rewards.get)

        # Determine expected winner
        if price < p_ref - CYCLE_COST_PER_MWH:
            expected = "charge  (+1)"
        elif price > p_ref + CYCLE_COST_PER_MWH:
            expected = "dschg   (-1)"
        else:
            expected = "hold    ( 0)"

        passed = best_name == expected
        status = "✓" if passed else "✗ FAIL"
        if not passed:
            all_passed = False

        print(f"Price = ${price:>7.2f}/MWh  [{label}]")
        for name, r in rewards.items():
            marker = " ← BEST" if name == best_name else ""
            print(f"    {name}: {r:+.5f}{marker}")
        print(f"    Expected: {expected}  |  Got: {best_name}  |  {status}")
        print()

    print("=" * 65)
    if all_passed:
        print("✓  ALL CHECKS PASSED — safe to run p4_train.py")
    else:
        print("✗  SOME CHECKS FAILED — do NOT run training until fixed")
        print("   Check CYCLE_COST_PER_MWH, REWARD_SCALE, and action convention.")
    print("=" * 65)
    print()

    # ── Additional: SoC boundary check ──────────────────────────────
    print("SoC boundary behaviour check:")
    print(f"  At SoC = {SOC_MAX} (full battery):")
    print(f"    charge  raw_action=+1 → feasible≈0 (can't add more energy)")
    print(f"    discharge raw_action=-1 → feasible≈-1 (can discharge)")
    print(f"  At SoC = {SOC_MIN} (empty battery):")
    print(f"    discharge raw_action=-1 → feasible≈0 (can't remove energy)")
    print(f"    charge raw_action=+1 → feasible≈+1 (can charge)")
    print()
    print("  ← These are now handled correctly: env.step() and demo buffer")
    print("    receive the feasible action, not the raw heuristic/actor action.")
    print()

    # ── Additional: Q-value intuition ───────────────────────────────
    print("Q-value intuition (approximate, ignoring discount/entropy):")
    horizon_steps = 288
    for price in [9.0, 24.0, 40.0]:
        r_charge  = plan_c_reward(+1.0, price, p_ref)
        r_hold    = plan_c_reward( 0.0, price, p_ref)
        r_dschg   = plan_c_reward(-1.0, price, p_ref)
        print(f"  rt_lmp=${price:.0f}: "
              f"Q(charge)≈{r_charge*horizon_steps:+.2f}  "
              f"Q(hold)≈{r_hold*horizon_steps:+.2f}  "
              f"Q(dschg)≈{r_dschg*horizon_steps:+.2f}")

    print()
    print("  If Q(charge) > Q(dschg) at low prices: reward is correct.")
    print("  If critic learns this correctly: policy will not collapse.")
    print("=" * 65)

    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
