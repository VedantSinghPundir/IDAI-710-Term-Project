"""
Pipeline 0 — Download Dataset from GitHub
==========================================
Downloads all monthly parquet files from:
  karthikmattu06-hue/hybridbid  →  data/processed/

Run this FIRST before anything else.

Usage:
    python pipeline/p0_download_data.py

Output:
    data/processed/energy_prices/2020-01.parquet  ...
    data/processed/as_prices/2020-01.parquet      ...
    data/processed/system_conditions/2020-01.parquet ...
"""

import os
import time
import requests

# ── CONFIG ────────────────────────────────────────────────────────────
GITHUB_API   = "https://api.github.com"
REPO         = "karthikmattu06-hue/hybridbid"
BASE_PATH    = "data/processed"
FOLDERS      = ["energy_prices", "as_prices", "system_conditions"]
SAVE_ROOT    = "./data/processed"          # local save location
DELAY_SEC    = 0.4                         # polite delay between requests
# If repo is private, paste your GitHub token here:
GITHUB_TOKEN = ""                          # e.g. "ghp_xxxxxxxxxxxx"
# ─────────────────────────────────────────────────────────────────────


def get_headers():
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


def list_files(folder: str) -> list[dict]:
    """Return list of {name, download_url} dicts for a folder."""
    url = f"{GITHUB_API}/repos/{REPO}/contents/{BASE_PATH}/{folder}"
    r = requests.get(url, headers=get_headers(), timeout=15)
    if r.status_code == 403:
        raise RuntimeError(
            "GitHub API rate limit hit or repo is private.\n"
            "Fix: Set GITHUB_TOKEN at the top of this file.\n"
            "Get a free token at: https://github.com/settings/tokens\n"
            "(No scopes needed for public repos)"
        )
    r.raise_for_status()
    return [
        {"name": f["name"], "download_url": f["download_url"]}
        for f in r.json()
        if f["name"].endswith(".parquet")
    ]


def download_file(download_url: str, save_path: str):
    """Download one file and save it locally."""
    r = requests.get(download_url, headers=get_headers(), timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(r.content)


def main():
    print("=" * 55)
    print("Pipeline 0 — Downloading ERCOT dataset from GitHub")
    print("=" * 55)

    total_downloaded = 0
    total_skipped    = 0

    for folder in FOLDERS:
        print(f"\n[{folder}] Fetching file list...")
        try:
            files = list_files(folder)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Found {len(files)} parquet files")
        folder_dir = os.path.join(SAVE_ROOT, folder)
        os.makedirs(folder_dir, exist_ok=True)

        for i, f in enumerate(files, 1):
            save_path = os.path.join(folder_dir, f["name"])

            if os.path.exists(save_path):
                size_kb = os.path.getsize(save_path) / 1024
                print(f"  [{i:3d}/{len(files)}] SKIP  {f['name']}  ({size_kb:.0f} KB already exists)")
                total_skipped += 1
                continue

            try:
                download_file(f["download_url"], save_path)
                size_kb = os.path.getsize(save_path) / 1024
                print(f"  [{i:3d}/{len(files)}] OK    {f['name']}  ({size_kb:.0f} KB)")
                total_downloaded += 1
                time.sleep(DELAY_SEC)
            except Exception as e:
                print(f"  [{i:3d}/{len(files)}] FAIL  {f['name']}  → {e}")

    print("\n" + "=" * 55)
    print(f"Done.  Downloaded: {total_downloaded}  |  Skipped: {total_skipped}")
    print(f"Data saved to: {os.path.abspath(SAVE_ROOT)}")
    print("\nNext step:  python pipeline/p1_inspect_data.py")


if __name__ == "__main__":
    main()
