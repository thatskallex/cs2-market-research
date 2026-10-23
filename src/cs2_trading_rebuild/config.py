from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["project"].get("root_dir", ".")).resolve()
    cfg["_config_path"] = str(path.resolve())
    cfg["_root"] = str(root)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    root = Path(cfg["_root"])
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else root / p


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
