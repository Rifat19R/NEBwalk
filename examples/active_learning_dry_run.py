"""Fast closed-loop orchestration sanity run with no MACE, QE, GPU, or network."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from nebwalk.dryrun import run_dry_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        result = run_dry_campaign(args.output)
        print(f"state={result.state.stage.value}")
        print(f"reason={result.state.stopping_reason}")
        print(f"campaign={result.campaign_dir}")
        return
    with tempfile.TemporaryDirectory(prefix="nebwalk-dry-run-") as temporary:
        result = run_dry_campaign(temporary)
        print(f"state={result.state.stage.value}")
        print(f"reason={result.state.stopping_reason}")
        print("sanity=passed")


if __name__ == "__main__":
    main()
