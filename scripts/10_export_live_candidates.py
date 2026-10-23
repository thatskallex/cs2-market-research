#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_trading_rebuild.config import load_config, resolve_path
from cs2_trading_rebuild.live import LIVE_CANDIDATE_INPUT_COLS, export_live_candidates
from cs2_trading_rebuild.utils import save_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    policy_panel_path = resolve_path(cfg, "signals_dir") / "policy_panel.parquet"
    policy_panel = pd.read_parquet(policy_panel_path, columns=LIVE_CANDIDATE_INPUT_COLS)

    policy = json.loads((resolve_path(cfg, "models_dir") / "policy.json").read_text(encoding="utf-8"))

    live_candidates = export_live_candidates(policy_panel, policy)

    live_dir = resolve_path(cfg, "live_dir")
    live_dir.mkdir(parents=True, exist_ok=True)
    save_table(live_candidates, live_dir / "live_candidates.parquet")

    print(f"Saved live candidates to {live_dir / 'live_candidates.parquet'}")


if __name__ == "__main__":
    main()