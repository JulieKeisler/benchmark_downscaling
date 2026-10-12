#!/usr/bin/env python3
"""Check which factorial plot_data.json have _events + doy."""
import json
import glob
import sys

root = sys.argv[1]
files = sorted(glob.glob(f"{root}/*/*/*/plot_data.json"))
with_events = with_doy = 0
missing = []
for f in files:
    try:
        d = json.load(open(f))
    except Exception as e:
        missing.append((f, f"parse:{e}"))
        continue
    has_ev = "_events" in d
    has_doy = any("doy" in (d.get(k) or {}) for k in d if k != "_events")
    with_events += has_ev
    with_doy += has_doy
    if not (has_ev and has_doy):
        missing.append((f, f"events={has_ev}, doy={has_doy}"))
print(f"{root}: {len(files)} plot_data.json | _events={with_events} | doy={with_doy}")
for f, why in missing:
    print(f"  MISS {f}  {why}")