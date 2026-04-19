"""
Pipeline 1 — Inspect & Validate Dataset
=========================================
Run this AFTER p0_download_data.py.
It reads one sample parquet from each folder and prints:
  - Column names  (copy these into config.py!)
  - Shapes, dtypes
  - Basic statistics
  - Missing value counts
  - Timestamp range and resolution check

Usage:
    python pipeline/p1_inspect_data.py

OUTPUT YOU NEED TO COPY:
    At the end it prints the exact column lists to paste
    into pipeline/config.py
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_ROOT = "./data/processed"
FOLDERS   = ["energy_prices", "as_prices", "system_conditions"]


def load_sample(folder: str) -> pd.DataFrame:
    pattern = os.path.join(DATA_ROOT, folder, "*.parquet")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquets found in {DATA_ROOT}/{folder}/\n"
                                "Did you run p0_download_data.py first?")
    # Load first file as sample
    return pd.read_parquet(files[0]), files[0]


def detect_timestamp_col(df: pd.DataFrame) -> str:
    """Find which column is the timestamp."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if "time" in col.lower() or "date" in col.lower() or "ts" == col.lower():
            return col
    # fallback: check if index is datetime
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return "__index__"
    return df.columns[0]


def main():
    print("=" * 65)
    print("Pipeline 1 — Dataset Inspection")
    print("=" * 65)

    all_cols = {}

    for folder in FOLDERS:
        print(f"\n{'─'*65}")
        print(f"FOLDER: {folder}")
        print(f"{'─'*65}")

        try:
            df, filepath = load_sample(folder)
            print(f"File   : {os.path.basename(filepath)}")
            print(f"Shape  : {df.shape}  ({df.shape[0]} rows × {df.shape[1]} cols)")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        # Timestamp detection
        ts_col = detect_timestamp_col(df)
        if ts_col == "__index__":
            ts = df.index
            print(f"Timestamp: index (datetime)")
        else:
            ts = pd.to_datetime(df[ts_col])
            print(f"Timestamp col: '{ts_col}'")
        print(f"Date range : {ts.min()} → {ts.max()}")
        # Make sure timestamp differences are handled as a Series, not TimedeltaIndex
        ts_sorted = pd.Series(ts).sort_values()
        diffs = ts_sorted.diff().dropna()
        if len(diffs) > 0:
            most_common_freq = diffs.mode().iloc[0]
            print(f"Most common interval: {most_common_freq}  (expected: 0 days 00:05:00)")
            if most_common_freq != pd.Timedelta("5min"):
              print("  ⚠ WARNING: interval is NOT 5 minutes — check this!")
        else:
            print("Most common interval: could not determine (not enough timestamps)")
        # print(f"Date range : {ts.min()} → {ts.max()}")
        # diffs = ts.sort_values().diff().dropna()
        # most_common_freq = diffs.mode()[0]
        # print(f"Most common interval: {most_common_freq}  (expected: 0 days 00:05:00)")
        # if most_common_freq != pd.Timedelta("5min"):
        #     print("  ⚠ WARNING: interval is NOT 5 minutes — check this!")
      

        # Column info
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        other_cols   = [c for c in df.columns if c not in numeric_cols and c != ts_col]

        print(f"\nAll columns ({len(df.columns)}):")
        for c in df.columns:
            dtype = df[c].dtype
            nulls = df[c].isnull().sum()
            print(f"  {c:<35} dtype={str(dtype):<12} nulls={nulls}")

        print(f"\nNumeric columns ({len(numeric_cols)}):")
        if numeric_cols:
            stats = df[numeric_cols].describe().round(3)
            print(stats.to_string())

        # Missing values
        total_nulls = df.isnull().sum().sum()
        print(f"\nTotal null values: {total_nulls}")

        all_cols[folder] = numeric_cols

    # ── Print the config block to copy ────────────────────────────────
    print("\n" + "=" * 65)
    print("ACTION REQUIRED — Copy these into pipeline/config.py")
    print("=" * 65)
    print()

    ep_cols = all_cols.get("energy_prices", [])
    as_cols = all_cols.get("as_prices", [])
    sc_cols = all_cols.get("system_conditions", [])

    # Guess which is which based on keywords
    price_candidates = []
    for c in ep_cols + as_cols:
        price_candidates.append(c)
    price_candidates = list(dict.fromkeys(price_candidates))  # deduplicate

    sys_candidates = [c for c in sc_cols
                      if c not in price_candidates]

    print("# In config.py, set these lists exactly:")
    print(f"ENERGY_PRICE_COLS  = {ep_cols}")
    print(f"AS_PRICE_COLS      = {as_cols}")
    print(f"SYSTEM_COLS        = {sc_cols}")
    print()
    print("# The 12-dim TTFE price vector should be (edit to match paper):")
    print(f"# [rt_lmp, 5×rt_mcpc, dam_spp, 5×dam_as]")
    print(f"# First 12 of: {price_candidates[:12]}")
    print()
    print("Next step:  Fill in config.py, then run  python pipeline/p2_build_dataset.py")


if __name__ == "__main__":
    main()
