#!/usr/bin/env python3
"""Fold a partial run of the net-load change lens into the full one.

When one method's predictions change (e.g. a re-decoded baseline), the change lens is rerun
with a future root holding only that method and the bicubic reference. This replaces that
method's entries (and its same-TAM check) in every results/tam_v2_change/metrics_change_*.json
with those of the partial run, after checking that the bicubic reference is unchanged.

Usage: python tools/merge_change_lens.py --partial results/tam_v2_change_sfc --method serpentflowclim
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--partial", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--full", default=str(ROOT / "results" / "tam_v2_change"))
    args = ap.parse_args()
    for part in sorted(Path(args.partial).glob("metrics_change_*.json")):
        full = Path(args.full) / part.name
        a, b = json.loads(full.read_text()), json.loads(part.read_text())
        ref_a, ref_b = a["methods"]["bicubic"], b["methods"]["bicubic"]
        assert ref_a == ref_b, f"{part.name}: bicubic reference differs"
        a["methods"][args.method] = b["methods"][args.method]
        a["check_w1_vs_lens"][args.method] = b["check_w1_vs_lens"][args.method]
        full.write_text(json.dumps(a, indent=1))
        print(f"{part.name}: {args.method} change W1 {b['methods'][args.method]['change_w1']:.1f}")


if __name__ == "__main__":
    main()
