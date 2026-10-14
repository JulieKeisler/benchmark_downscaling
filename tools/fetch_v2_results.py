#!/usr/bin/env python3
"""Copy the finished v2 evaluation outputs from Jean-Zay and cleps to a local tree.

For every (method, region, GCM) whose latest v2 job (seed 0 / member 1) has
COMPLETED, rsyncs <cluster results_v2>/<sub>/<method dir>/ to
<out>/<sub>/<method dir>/, without predictions or checkpoints; <sub> is
<region> for the region's default GCM, <region>/<gcm> otherwise (the
dashboard layout). Older local copies of the same dir are replaced. A cell
whose latest job is not finished is left out, so a partial tree never mixes
runs.

Usage: python tools/fetch_v2_results.py [--out results/v2]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_status as vs  # noqa: E402

ROOTS = {"jz": "/lustre/fswork/projects/rech/bqu/uyv87oi/benchmark_downscaling/results_v2",
         "cleps": "/scratch/jkeisler/benchmark_downscaling/results_v2"}
DIRS = {"bicubic": "bicubic", "eqm": "eqm", "bcsd": "bcsd", "cdft": "cdft", "dotc": "dotc",
        "idownscale": "idownscale", "corrdiff": "corrdiff_qm", "cm": "cm", "sfc": "serpentflowclim"}


def finished_cells():
    jobs = {}
    for host in ("jz", "cleps"):
        jobs.update({f"{host}:{k}": v for k, v in vs.fetch(host).items()})
    latest = {}
    for j in jobs.values():
        meth, region, gcm, seed, _ = j["key"]
        if meth not in DIRS or seed not in (None, 0):
            continue
        k = (meth, region, gcm)
        if k not in latest or j["submit"] > latest[k]["submit"]:
            latest[k] = j
    return {k: j for k, j in latest.items() if j["state"] == "COMPLETED"}, latest


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "v2"))
    args = p.parse_args()
    out = Path(args.out)
    done, latest = finished_cells()
    print(f"{len(done)} finished cells of {len(latest)} with a job")
    for (meth, region, gcm), j in sorted(done.items()):
        sub = region if gcm == vs.NATIVE[region] else f"{region}/{gcm}"
        src = f"{j['host']}:{ROOTS[j['host']]}/{sub}/{DIRS[meth]}/"
        dst = out / sub / DIRS[meth]
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        subprocess.run(["rsync", "-a", "--exclude=predictions.pt", "--exclude=checkpoints/",
                        "--exclude=*.pth", src, str(dst) + "/"], check=True)
        if not (dst / "metrics.json").exists():
            raise SystemExit(f"{src}: finished job {j['id']} but no metrics.json")
        print(f"  {sub}/{DIRS[meth]} <- {j['host']} job {j['id']}")
    # dirs of cells that are not finished any more (re-submitted) must not stay
    keep = {(region if gcm == vs.NATIVE[region] else f"{region}/{gcm}", DIRS[m]) for m, region, gcm in done}
    for mj in out.glob("**/metrics.json"):
        rel = mj.parent.relative_to(out)
        if (str(rel.parent), rel.name) not in keep:
            shutil.rmtree(mj.parent)
            print(f"  removed stale {rel}")


if __name__ == "__main__":
    main()
