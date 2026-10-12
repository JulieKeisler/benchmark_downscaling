#!/usr/bin/env python3
import json, glob, sys
NATIVE_GCM = {
    "australia": ("cesm2", "ipsl_cm6a_lr"),
    "france": ("cesm2", "cnrm_cm6_1"),
    "us_central": ("cnrm_cm6_1", "ipsl_cm6a_lr"),
}
root = sys.argv[1]
ok = miss = 0
for reg, gcms in NATIVE_GCM.items():
    for g in gcms:
        files = sorted(glob.glob(f"{root}/{reg}/{g}/*/plot_data.json"))
        row = [f"{reg}/{g}:"]
        for f in files:
            d = json.load(open(f))
            ev = "_events" in d
            doy = any("doy" in (d.get(k) or {}) for k in d if k != "_events")
            m = f.split("/")[-2]
            if ev and doy:
                ok += 1
                row.append(f"{m}ok")
            else:
                miss += 1
                row.append(f"{m}MISS")
        if files:
            print(" ".join(row))
print(f"TOTAL ok={ok} misses={miss}")