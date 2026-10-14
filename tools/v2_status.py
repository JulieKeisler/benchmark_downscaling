#!/usr/bin/env python3
"""v2 campaign status table (GCM x region rows, method columns) as a PDF.

Reads Slurm accounting and queues on Jean-Zay and cleps over ssh (read-only),
keeps the latest submission per (method, region, GCM, seed), and writes a
landscape PDF: done / running (where, expected end) / pending (where,
expected start).

Expected end of a running job = its start + the median duration of finished
v2 jobs of the same kind (a default when fewer than two have finished).
Expected start of a pending job = Slurm's estimate (squeue --start), or the
expected end of the job it depends on.

Usage: python tools/v2_status.py [--out v2_status.pdf]
"""
from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

SINCE = "2026-10-13T00:00"
REGIONS = {"france": "France", "australia": "Australia", "us_central": "Central US"}
GCMS = {"ipsl_cm6a_lr": "IPSL-CM6A-LR", "cnrm_cm6_1": "CNRM-CM6-1", "cesm2": "CESM2"}
NATIVE = {"france": "ipsl_cm6a_lr", "australia": "cnrm_cm6_1", "us_central": "cesm2"}
METHODS = [("bicubic", "Bicubic"), ("eqm", "EQM"), ("bcsd", "BCSD"), ("cdft", "CDF-t"),
           ("dotc", "dOTC"), ("idownscale", "iDownscale"), ("corrdiff", "CorrDiff-QM"),
           ("cm", "CM"), ("sfc", "SerpentFlow-CLIM")]
SEEDS = {"cm": [0, 1, 2], "corrdiff": [0, 1, 2]}
# CorrDiff seed 0 trains a model for these pairs; the others reuse one (prediction only).
CORRDIFF_TRAINS = {("france", "ipsl_cm6a_lr"), ("australia", "cnrm_cm6_1"), ("us_central", "cesm2"),
                   ("france", "cesm2"), ("australia", "cesm2"), ("us_central", "ipsl_cm6a_lr")}
# Hours, used until two jobs of the kind have finished (measured on 13-14 Oct runs).
DEFAULT_H = {"bicubic": 0.5, "eqm": 0.5, "bcsd": 0.6, "cdft": 4.2, "dotc": 2.3, "idownscale": 2.5,
             "corrdiff/train": 2.9, "corrdiff/predict": 1.5, "corrdiff/seed": 1.6, "cm": 0.45,
             "sfc/fresh": 11.0, "sfc/cached": 1.0, "psd": 0.4}
NAME_RE = re.compile(r"^v2_(?P<m>.+?)_(?P<r>france|australia|us_central)"
                     r"(?:_(?P<g>ipsl_cm6a_lr|cnrm_cm6_1|cesm2))?(?:_s(?P<s>\d))?$")
MOVER = "/lustre/fswork/projects/rech/bqu/uyv87oi/benchmark_downscaling/logs/v2/devmover.sh"


def ssh(host: str, cmd: str) -> str:
    return subprocess.run(["ssh", "-o", "ConnectTimeout=30", host, cmd], capture_output=True,
                          text=True, timeout=120, check=True).stdout


def parse_time(s: str):
    if not s or s in ("Unknown", "N/A", "None", "(null)"):
        return None
    return datetime.fromisoformat(s)


def limit_minutes(s: str):
    m = re.match(r"(?:(\d+)-)?(\d+):(\d+):(\d+)$", s)
    return (int(m[1] or 0) * 1440 + int(m[2]) * 60 + int(m[3])) if m else None


def parse_name(name: str):
    m = NAME_RE.match(name)
    if not m:
        return None
    meth, seed, variant = m["m"], int(m["s"]) if m["s"] else None, None
    if meth.startswith("reeval_"):
        meth, variant = meth[len("reeval_"):], "reeval"
    if meth.startswith("eval_"):   # evaluation only, on existing predictions
        meth, variant = meth[len("eval_"):], "evalonly"
        sm = re.match(r"(.+)_s(\d)$", meth)
        if sm:
            meth, seed = sm[1], int(sm[2])
    if meth == "corrdiff_qm":
        meth = "corrdiff"
    region, gcm = m["r"], m["g"] or NATIVE[m["r"]]
    if meth in SEEDS and seed is None:
        seed = 0
    if variant is None:
        if meth == "corrdiff":
            variant = "seed" if seed else ("train" if (region, gcm) in CORRDIFF_TRAINS else "predict")
        elif meth == "sfc":
            variant = "cached" if gcm == NATIVE[region] and region != "us_central" else "fresh"
    return meth, region, gcm, seed, variant


def fetch(host: str):
    acct = ssh(host, f"sacct -u $USER -S {SINCE} -X -n -P -o "
                     "JobID,JobName,Partition,QOS,Account,State,Submit,Start,End,ElapsedRaw,TimelimitRaw,NodeList")
    queue = ssh(host, "squeue -h -u $USER -o '%i|%T|%S|%E|%P|%q|%a|%N|%j|%V|%l'")
    starts = ssh(host, "squeue -h -u $USER -t PD --start -o '%i|%S'")
    jobs = {}
    for line in acct.splitlines():
        f = line.split("|")
        if len(f) < 12 or not f[1].startswith("v2_"):
            continue
        key = parse_name(f[1])
        if key is None:
            continue
        jobs[f[0]] = dict(host=host, id=f[0], name=f[1], key=key, partition=f[2], qos=f[3], account=f[4],
                          state=f[5].split()[0], submit=parse_time(f[6]), start=parse_time(f[7]),
                          end=parse_time(f[8]), elapsed=int(f[9] or 0),
                          limit_min=int(f[10]) if f[10].isdigit() else None, node=f[11], dep=None)
    for line in queue.splitlines():
        i, st, start, dep, part, qos, account, node, name, submit, limit = line.split("|")
        # pending jobs held by a dependency are not in sacct's time window yet
        if i not in jobs and name.startswith("v2_") and parse_name(name):
            jobs[i] = dict(host=host, id=i, name=name, key=parse_name(name), submit=parse_time(submit),
                           start=None, end=None, elapsed=0, limit_min=limit_minutes(limit))
        if i in jobs:
            j = jobs[i]
            j.update(state=st, partition=part, qos=qos, account=account, node=node)
            j["start"] = parse_time(start) if st == "RUNNING" else j["start"]
            deps = re.findall(r"after\w*:(\d+)", dep) if dep and dep != "(null)" else []
            j["dep"] = deps or None
    for line in starts.splitlines():
        i, s = line.split("|")
        if i in jobs:
            jobs[i]["est_start"] = parse_time(s)
    return jobs


def typical_hours(jobs) -> dict:
    by_kind = {}
    for j in jobs.values():
        meth, *_, variant = j["key"]
        kind = f"{meth}/{variant}" if variant else meth
        if j["state"] == "COMPLETED" and j["elapsed"] > 120:
            by_kind.setdefault(kind, []).append(j["elapsed"] / 3600)
    out = dict(DEFAULT_H)
    for kind, hs in by_kind.items():
        if len(hs) >= 2:
            out[kind] = statistics.median(hs)
    return out


def kind_of(j):
    meth, *_, variant = j["key"]
    return f"{meth}/{variant}" if variant else meth


def where(j) -> str:
    if j["host"] == "jz":
        q = "dev" if "dev" in j["qos"] else ("t3" if "t3" in j["qos"] else j["qos"])
        s = f"JZ {j['partition']} {q}"
    else:
        s = f"cleps {j['partition']}" + (" (arches)" if j["account"] == "arches" else "")
    if j["state"] == "RUNNING" and j["node"]:
        s += f", {j['node']}"
    return tex(s)


def fmt(t: datetime, now: datetime) -> str:
    if t.date() == now.date():
        return t.strftime("%H:%M")
    return t.strftime("%d %b %H:%M").lstrip("0")


def tex(s: str) -> str:
    return s.replace("\\", "/").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


class Status:
    def __init__(self, jobs, typical, mover_ids, now):
        self.jobs, self.typ, self.mover, self.now = jobs, typical, mover_ids, now
        self.by_id = {j["id"]: j for j in jobs.values()}

    def expected_end(self, j, depth=0):
        h = timedelta(hours=self.typ.get(kind_of(j), 1.0))
        if j["state"] == "COMPLETED":
            return j["end"]
        if j["state"] == "RUNNING" and j["start"]:
            return max(j["start"] + h, self.now + timedelta(minutes=10))
        s = self.expected_start(j, depth)
        return s + h if s else None

    def expected_start(self, j, depth=0):
        if j.get("dep") and depth < 5:
            ends = [self.expected_end(self.by_id[d], depth + 1) for d in j["dep"] if d in self.by_id]
            ends = [e for e in ends if e]
            return max(ends) if ends else None
        return j.get("est_start")

    def line(self, j) -> tuple[str, str]:
        st = j["state"]
        if st == "COMPLETED":
            v = j["key"][-1]
            note = " (v1 output re-scored)" if v == "reeval" else (" (re-scored)" if v == "evalonly" else "")
            return "done", f"done {fmt(j['end'], self.now)}{note}"
        if st == "RUNNING":
            e = self.expected_end(j)
            late = j["start"] + timedelta(hours=self.typ.get(kind_of(j), 1.0)) < self.now
            lim = j["start"] + timedelta(minutes=j["limit_min"]) if j["limit_min"] else None
            txt = f"running, {where(j)}; ends " + ("soon (past typical)" if late else f"$\\sim${fmt(e, self.now)}")
            if late and lim:
                txt += f", limit {fmt(lim, self.now)}"
            return "running", txt
        if st == "PENDING":
            if j.get("dep"):
                dep = self.by_id.get(j["dep"][0])
                what = dep["key"][0] if dep else "job " + j["dep"][0]
                if dep and dep["key"][0] == "corrdiff":
                    what = "the CorrDiff model " + ("training" if dep["key"][-1] == "train" else "prediction")
                if dep and dep["key"][1:3] != j["key"][1:3]:
                    what += f" of {REGIONS[dep['key'][1]]}$\\times${GCMS[dep['key'][2]].split('-')[0]}"
                s = self.expected_start(j)
                return "pending", f"pending, after {what}" + (f" ($\\sim${fmt(s, self.now)})" if s else "") + f"; {where(j)}"
            if j["id"] in self.mover and "dev" not in j["qos"] and j["host"] == "jz":
                return "pending", "pending, goes to JZ gpu\\_p13 dev as a slot frees (within $\\sim$1 h)"
            s = j.get("est_start")
            return "pending", f"pending, {where(j)}; starts " + (f"$\\sim${fmt(s, self.now)}" if s else "(no Slurm estimate yet)")
        return "failed", f"{st.lower().replace('_', ' ')} (job {j['id']}, {where(j)})"


def build(jobs, status: Status):
    latest = {}
    for j in jobs.values():
        meth, region, gcm, seed, _ = j["key"]
        k = (meth, region, gcm, seed)
        if k not in latest or (j["submit"] or datetime.min) > (latest[k]["submit"] or datetime.min):
            latest[k] = j
    cells = {}
    for region in REGIONS:
        for gcm in GCMS:
            for meth, _ in METHODS:
                seeds = SEEDS.get(meth, [None])
                lines = []
                for s in seeds:
                    j = latest.get((meth, region, gcm, s))
                    st, txt = status.line(j) if j else ("missing", "not submitted")
                    lab = "" if s is None else (f"s{s}: " if meth == "cm" else
                                                (("model: " if s == 0 else f"s{s}: ")))
                    lines.append((st, lab, txt))
                if meth == "sfc":
                    lines = [(st, "3 members: ", txt) for st, _, txt in lines]
                states = [x[0] for x in lines]
                if all(x == "done" for x in states) and len(lines) > 1:
                    lines = [("done", "", f"done ({len(lines)} seeds)")]
                elif meth == "cm" and len({x[2] for x in lines}) == 1:
                    lines = [(lines[0][0], "s0--s2: ", lines[0][2])]
                elif meth == "corrdiff" and len(lines) == 3 and lines[1][2] == lines[2][2]:
                    lines = [lines[0], (lines[1][0], "s1, s2: ", lines[1][2])]
                if any(x in ("failed", "missing") for x in states):
                    cs = "failed"
                elif all(x == "done" for x in states):
                    cs = "done"
                elif any(x == "running" for x in states):
                    cs = "running"
                else:
                    cs = "pending"
                cells[(region, gcm, meth)] = (cs, lines)
    psd = [latest.get(("psd", r, g, None)) for r in REGIONS for g in GCMS]
    n_psd = sum(1 for j in psd if j and j["state"] == "COMPLETED")
    return cells, n_psd


def render(cells, n_psd, now, out: Path, typ):
    color = {"done": "done", "running": "run", "pending": "pend", "failed": "fail"}
    rows = []
    for gcm, glab in GCMS.items():
        for i, (region, rlab) in enumerate(REGIONS.items()):
            first = f"\\multirow{{3}}{{*}}{{\\textbf{{{glab}}}}}" if i == 0 else ""
            rcell = rlab
            cols = [first, rcell]
            for meth, _ in METHODS:
                cs, lines = cells[(region, gcm, meth)]
                body = " \\newline ".join(f"{lab}{txt}" for _, lab, txt in lines)
                cols.append(f"\\cellcolor{{{color[cs]}}}{body}")
            rows.append(" & ".join(cols) + " \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    counts = {}
    for cs, lines in cells.values():
        counts[cs] = counts.get(cs, 0) + 1
    head = " & ".join(["\\textbf{GCM}", "\\textbf{Region}"] + [f"\\textbf{{{m}}}" for _, m in METHODS])
    w = ">{\\raggedright\\arraybackslash}p{3.38cm}"
    doc = rf"""\documentclass{{article}}
\usepackage[a3paper,landscape,margin=1.2cm]{{geometry}}
\usepackage[table]{{xcolor}}
\usepackage{{multirow,booktabs,makecell,array}}
\usepackage[T1]{{fontenc}}\usepackage{{lmodern}}
\definecolor{{done}}{{HTML}}{{E3F1E0}}\definecolor{{run}}{{HTML}}{{DCE9F7}}
\definecolor{{pend}}{{HTML}}{{FBEFD5}}\definecolor{{fail}}{{HTML}}{{F6D5D5}}
\renewcommand{{\arraystretch}}{{1.25}}
\pagestyle{{empty}}
\begin{{document}}
{{\Large\textbf{{Benchmark v2 runs: status}}}}\hfill generated {now.strftime('%d %b %Y %H:%M')} (CEST)\par\medskip
\noindent Cells: {counts.get('done', 0)} done, {counts.get('running', 0)} running, {counts.get('pending', 0)} pending, {counts.get('failed', 0)} failed (of {len(cells)}).
\quad\colorbox{{done}}{{done}}\ \colorbox{{run}}{{running}}\ \colorbox{{pend}}{{pending}}\ \colorbox{{fail}}{{failed}}\par\medskip
\footnotesize
\begin{{tabular}}{{>{{\raggedright\arraybackslash}}p{{2.5cm}}>{{\raggedright\arraybackslash}}p{{1.8cm}}*{{9}}{{{w}}}}}
\toprule
{head} \\
\midrule
{chr(10).join(rows)}
\end{{tabular}}
\par\medskip
\noindent\begin{{minipage}}{{\textwidth}}\footnotesize
\textbf{{Reading the table.}} CorrDiff-QM: ``model'' is seed 0 (it trains a model, or reuses one trained on the same region and variables), s1--s2 the extra seeds.
CM: seeds 0--2. SerpentFlow-CLIM: one job writes members 1--3. Times are today unless a date is shown.
Expected end of a running job = start + median duration of finished v2 jobs of the same kind
(CDF-t {typ['cdft']:.1f} h, dOTC {typ['dotc']:.1f} h, iDownscale {typ['idownscale']:.1f} h, CorrDiff training {typ['corrdiff/train']:.1f} h,
CorrDiff prediction {typ['corrdiff/predict']:.1f} h, CorrDiff seed {typ['corrdiff/seed']:.1f} h, CM {typ['cm']:.2f} h, SerpentFlow-CLIM {typ['sfc/fresh']:.0f} h on V100).
Expected start of a pending job = Slurm's estimate, or the expected end of the job it waits for.
Not in the table: t* sweeps for CM, {n_psd}/9 done; CM t* sensitivity, done.
\end{{minipage}}
\end{{document}}
"""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "s.tex").write_text(doc)
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "s.tex"], cwd=d,
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stdout[-3000:])
        shutil.copy(Path(d) / "s.pdf", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "v2_status.pdf"))
    args = p.parse_args()
    jobs = {}
    for host in ("jz", "cleps"):
        jobs.update({f"{host}:{k}": v for k, v in fetch(host).items()})
    # job ids are cluster-local but JZ (10xxxxx) and cleps (55xxxxx) ranges do not overlap
    mover = set(re.findall(r"\d{7}", ssh("jz", f"grep -E '^(CM|CD)=' {MOVER} 2>/dev/null || true")))
    now = datetime.now().replace(microsecond=0)
    typ = typical_hours(jobs)
    status = Status(jobs, typ, mover, now)
    cells, n_psd = build(jobs, status)
    render(cells, n_psd, now, Path(args.out), typ)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
