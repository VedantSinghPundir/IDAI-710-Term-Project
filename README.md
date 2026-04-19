# Stage 1 — Pre-RTC+B Battery Bidding (SAC + TTFE)

## Project Structure

```
stage1/
├── pipeline/
│   ├── config.py            ← EDIT THIS with your column names
│   ├── p0_download_data.py  ← Step 1: download dataset from GitHub
│   ├── p1_inspect_data.py   ← Step 2: identify column names
│   ├── p2_build_dataset.py  ← Step 3: merge + validate + fit normaliser
│   ├── p3_models.py         ← Step 4: verify model shapes
│   ├── p4_train.py          ← Step 5: SAC training loop
│   └── p5_evaluate.py       ← Step 6: baselines + results table
├── data/
│   └── processed/           ← downloaded here by p0
│       ├── energy_prices/
│       ├── as_prices/
│       └── system_conditions/
├── checkpoints/
│   └── stage1/              ← model weights saved here
└── logs/                    ← training_log.csv, eval_results.csv
```

---

## Install Requirements

```bash
pip install torch pandas numpy pyarrow requests scipy
```

---

## Step-by-Step Instructions

### Step 1 — Download the dataset

```bash
python pipeline/p0_download_data.py
```

This downloads all monthly parquet files from:
`karthikmattu06-hue/hybridbid/data/processed/`

**If you get a 403 error**, the repo may be private or rate-limited.
Fix: go to https://github.com/settings/tokens, create a free token
(no scopes needed for public repos), and paste it into `p0_download_data.py`:

```python
GITHUB_TOKEN = "ghp_your_token_here"
```

Expected output:
```
[energy_prices]  Found 72 parquet files
  [  1/72] OK    2020-01.parquet  (45 KB)
  ...
Done.  Downloaded: 216  |  Skipped: 0
```

---

### Step 2 — Inspect column names

```bash
python pipeline/p1_inspect_data.py
```

This reads one sample parquet from each folder and prints the exact
column names. **Copy the output into `pipeline/config.py`.**

Expected output (example — your actual names may differ):
```
ENERGY_PRICE_COLS  = ['timestamp', 'rt_lmp', 'dam_spp']
AS_PRICE_COLS      = ['timestamp', 'rt_reg_up', 'rt_reg_dn', ...]
SYSTEM_COLS        = ['timestamp', 'load_fcst', 'load_act', 'wind_gen', ...]
```

**Action: open `pipeline/config.py` and update:**
```python
PRICE_COLS  = [...]   # 12 columns exactly
SYSTEM_COLS = [...]   # 7 columns exactly
TIMESTAMP_COL = "..."  # or "__index__"
```

---

### Step 3 — Build and validate the dataset

```bash
python pipeline/p2_build_dataset.py
```

This merges the three folders, validates columns, fits the normaliser,
and saves `checkpoints/stage1/normaliser_stats.npz`.

Expected output:
```
[p2] Merged: 633,856 rows | 19 columns
[p2] ✓ All required columns present
[p2] Train split: 603,000 rows
[p2] Val split  : 30,000 rows
[p2] Normaliser saved → checkpoints/stage1/normaliser_stats.npz
```

**If you see column mismatch errors**, go back to Step 2 and fix `config.py`.

---

### Step 4 — Verify model shapes

```bash
python pipeline/p3_models.py
```

Expected output:
```
  TTFE output  : (4, 64)         ← should be (4, 64)
  obs          : (4, 78)         ← should be (4, 78)
  action       : (4, 1)          ← should be (4, 1)
  TTFE params  : 92,224
  Actor params : 149,505
  Total        : 389,473
✓ All shapes correct.
```

---

### Step 5 — Train the agent

```bash
python pipeline/p4_train.py
```

This runs 500,000 SAC steps. On CPU it takes ~4-6 hours.
On GPU (if available) it takes ~45-90 minutes.

Checkpoints are saved every 50,000 steps to `checkpoints/stage1/`.
The best checkpoint (by val revenue) is saved as `stage1_best.pt`.

Training log is saved to `logs/training_log.csv`.

**To resume from a checkpoint**, add at the top of `main()`:
```python
start_step = agent.load("checkpoints/stage1/stage1_step200000.pt")
```

---

### Step 6 — Evaluate results

```bash
python pipeline/p5_evaluate.py
```

Expected output (numbers will vary):
```
  Method                              Total Rev ($)  vs Heuristic  % of PIO
  ─────────────────────────────────────────────────────────────────────────
  Rule-based heuristic               $     8,241.50          ---    41.2%
  PIO (perfect foresight LP)         $    20,003.80       +142.7%  100.0%
  SAC agent (step 500000)            $    13,105.20        +59.0%   65.5%
```

Share this table with your teammates as your Stage 1 result.

---

## What to Send to Your Teammates (for Stage 2)

The Stage 2 fine-tuning will load your Stage 1 TTFE weights.
Point them to:

```
checkpoints/stage1/stage1_best.pt
checkpoints/stage1/normaliser_stats.npz
```

The checkpoint contains keys: `ttfe`, `actor`, `critic`, `critic_tgt`, `log_alpha`.
Stage 2 will freeze `ttfe` initially, then selectively unfreeze it.

---

## Quick Troubleshooting

| Error | Fix |
|-------|-----|
| `No parquet files found` | Run `p0_download_data.py` first |
| `column mismatch` | Run `p1_inspect_data.py`, update `config.py` |
| `normaliser not found` | Run `p2_build_dataset.py` first |
| `No checkpoint found` | Run `p4_train.py` first |
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 128 in `config.py` |
| `403 from GitHub` | Add `GITHUB_TOKEN` in `p0_download_data.py` |
