from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import build_report, experiments, figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all PRBI experiments and build the final report.")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument(
        "--stage",
        choices=["setup", "data", "experiments", "figures", "report", "check", "all"],
        default="all",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = ["setup", "data", "experiments", "figures", "report", "check"] if args.stage == "all" else [args.stage]

    if "setup" in stages:
        experiments.record_environment(ROOT, args.device, args.seed)
    if "data" in stages:
        experiments.prepare_data(ROOT, args.profile)
    if "experiments" in stages:
        experiments.run_all(ROOT, args.profile, args.device, args.seed)
    if "figures" in stages:
        figures.generate_all(ROOT)
    if "report" in stages:
        build_report.build(ROOT)
    if "check" in stages:
        build_report.check(ROOT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
