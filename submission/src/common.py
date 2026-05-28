from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def submission(root: Path) -> Path:
    return root / "submission"


def results_dir(root: Path) -> Path:
    path = submission(root) / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(root: Path) -> Path:
    path = submission(root) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(root: Path) -> Path:
    path = submission(root) / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_dir(root: Path) -> Path:
    path = submission(root) / "report"
    path.mkdir(parents=True, exist_ok=True)
    (path / "build").mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_text(command: list[str], cwd: Path | None = None, timeout: int = 60) -> str:
    try:
        cp = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, encoding="utf-8", errors="replace")
        return (cp.stdout + cp.stderr).strip()
    except Exception as exc:
        return repr(exc)


def choose_device(mode: str):
    import torch

    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(mode)


def torch_home(root: Path) -> None:
    os.environ.setdefault("TORCH_HOME", str((root / "models").resolve()))
