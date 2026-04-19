"""
Pipeline 2 — Build & Validate Merged Dataset
=============================================
Merges the three parquet folders into one clean DataFrame,
fits the normaliser, and runs validation checks.

Run AFTER:
  p0_download_data.py  (download)
  p1_inspect_data.py   (identify column names → update config.py)

Usage:
    python pipeline/p2_build_dataset.py

Outputs:
    checkpoints/stage1/normaliser_stats.npz   ← mean/std saved for reuse
    (prints validation summary to console)
"""

import os
import sys
import glob
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.config import *


# ════════════════════════════════════════════════════════
# LOAD & MERGE
# ════════════════════════════════════════════════════════

def load_folder(subfolder: str) -> pd.DataFrame:
    """Load all monthly parquets in a subfolder into one DataFrame."""
    pattern = os.path.join(DATA_ROOT, subfolder, "*.parquet")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found at {pattern}\n"
            "Run p0_download_data.py first."
        )
    parts = [pd.read_parquet(f) for f in files]
    df    = pd.concat(parts, ignore_index=True)

    # Ensure datetime index
    if TIMESTAMP_COL == "__index__":
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
    else:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
        df = df.set_index(TIMESTAMP_COL)

    return df.sort_index()


def build_merged_df() -> pd.DataFrame:
    print("[p2] Loading energy_prices ...")
    energy  = load_folder("energy_prices")
    print(f"     {len(energy):,} rows | cols: {energy.columns.tolist()}")

    print("[p2] Loading as_prices ...")
    as_pr   = load_folder("as_prices")
    print(f"     {len(as_pr):,} rows | cols: {as_pr.columns.tolist()}")

    print("[p2] Loading system_conditions ...")
    syscond = load_folder("system_conditions")
    print(f"     {len(syscond):,} rows | cols: {syscond.columns.tolist()}")

    print("[p2] Merging on timestamp index ...")
    df = energy.join(as_pr,   how="outer", rsuffix="_as")
    df = df.join(syscond,     how="outer", rsuffix="_sys")
    df = df.sort_index()

    # Forward-fill small gaps (up to 3 intervals = 15 min)
    df = df.ffill(limit=3).dropna()

    # Clip to Stage 1 date range
    df = df[(df.index >= STAGE1_START) & (df.index <= STAGE1_END)]

    print(f"     Merged: {len(df):,} rows | {df.shape[1]} columns")
    print(f"     Date range: {df.index.min()} → {df.index.max()}")
    return df


# ════════════════════════════════════════════════════════
# COLUMN VALIDATION
# ════════════════════════════════════════════════════════

def validate_columns(df: pd.DataFrame):
    missing_price  = [c for c in PRICE_COLS  if c not in df.columns]
    missing_system = [c for c in SYSTEM_COLS if c not in df.columns]

    if missing_price:
        print("\n⚠ MISSING PRICE COLUMNS:")
        for c in missing_price:
            print(f"  '{c}' not found in merged DataFrame")
        print(f"\n  Available columns: {df.columns.tolist()}")
        print("\n  FIX: Update PRICE_COLS in pipeline/config.py to match")
        print("       the exact column names shown above, then re-run.\n")
        raise ValueError("Price column mismatch — update config.py")

    if missing_system:
        print("\n⚠ MISSING SYSTEM COLUMNS:")
        for c in missing_system:
            print(f"  '{c}' not found in merged DataFrame")
        raise ValueError("System column mismatch — update config.py")

    print("[p2] ✓ All required columns present")


# ════════════════════════════════════════════════════════
# FIT NORMALISER
# ════════════════════════════════════════════════════════

def fit_normaliser(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std on TRAIN split only (before VAL_START)."""
    train_df = df[df.index < VAL_START]
    all_cols = PRICE_COLS + SYSTEM_COLS
    mean = train_df[all_cols].mean().values.astype(np.float32)
    std  = train_df[all_cols].std().values.astype(np.float32)
    std  = np.where(std == 0, 1.0, std)   # avoid divide-by-zero
    return mean, std


# ════════════════════════════════════════════════════════
# DATASET STATISTICS REPORT
# ════════════════════════════════════════════════════════

def print_stats(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
    all_cols = PRICE_COLS + SYSTEM_COLS

    print("\n[p2] Dataset statistics (price columns):")
    print(f"  {'Column':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Skew':>8}")
    print(f"  {'-'*68}")
    for i, col in enumerate(PRICE_COLS):
        s = df[col]
        print(f"  {col:<20} {s.mean():>10.2f} {s.std():>10.2f} "
              f"{s.min():>10.2f} {s.max():>10.2f} {s.skew():>8.2f}")

    # Check for price spikes
    rt_lmp = df[PRICE_COLS[0]]
    spike_pct = (rt_lmp > 200).mean() * 100
    print(f"\n  RT LMP > $200/MWh : {spike_pct:.3f}% of intervals")
    print(f"  RT LMP max        : ${rt_lmp.max():,.1f}")

    # Train / val split sizes
    train_df = df[df.index < VAL_START]
    val_df   = df[df.index >= VAL_START]
    print(f"\n[p2] Train split : {len(train_df):,} rows  ({train_df.index.min().date()} → {train_df.index.max().date()})")
    print(f"[p2] Val split   : {len(val_df):,} rows  ({val_df.index.min().date()} → {val_df.index.max().date()})")
    print(f"[p2] Usable windows (after WINDOW_LEN={WINDOW_LEN} warmup): "
          f"{max(0, len(train_df) - WINDOW_LEN):,} train  |  {max(0, len(val_df) - WINDOW_LEN):,} val")

    # Null check after merge
    nulls = df[all_cols].isnull().sum()
    if nulls.sum() > 0:
        print(f"\n  ⚠ Null values after merge:")
        print(nulls[nulls > 0])
    else:
        print("\n[p2] ✓ No null values after merge and forward-fill")


# ════════════════════════════════════════════════════════
# SAVE NORMALISER
# ════════════════════════════════════════════════════════

def save_normaliser(mean: np.ndarray, std: np.ndarray):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, "normaliser_stats.npz")
    np.savez(path, mean=mean, std=std,
             price_cols=PRICE_COLS, system_cols=SYSTEM_COLS)
    print(f"\n[p2] Normaliser saved → {path}")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Pipeline 2 — Build & Validate Dataset")
    print("=" * 60)

    df = build_merged_df()
    validate_columns(df)
    mean, std = fit_normaliser(df)
    print_stats(df, mean, std)
    save_normaliser(mean, std)

    print("\n" + "=" * 60)
    print("✓ Pipeline 2 complete.")
    print("Next step:  python pipeline/p3_models.py  (verify models load)")
    print("Then:       python pipeline/p4_train.py")


if __name__ == "__main__":
    main()
