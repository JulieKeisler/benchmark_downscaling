# Scientific contract — Downscaling Benchmark for the Electricity System

This file is the human-authored **contract** that constrains agent-assisted
development of this repository. It fixes the quantities, units, invariants and
acceptance criteria that any change must respect. It is the analogue of the
"project constitution" in the legoESM human–agent workflow.

> Agents implement and test; **humans** own this contract and adjudicate
> scientific choices. If a change requires violating a clause here, the clause
> is amended first (by a human) — never silently worked around in code.

The mechanical enforcement lives in:

| Clause | Enforced by |
| --- | --- |
| §Units | `tests/test_plot_meta.py`, `src/dashboard/plot_meta.py` |
| §Plot labelling | `tests/test_plot_meta.py` |
| §Data invariants | `src/baselines/*` post-correction clipping, `tests/` |
| §Time alignment | `align_time`, re-stamp after `bicubic_regrid` (AGENTS.md gotcha 1) |
| §Registry | `tests/test_registry.py` |
| §Predictions format | `save_predictions` + `tests/test_methods.py` |

---

## 1. Purpose

A fair, reproducible comparison of downscaling / bias-correction / generative
methods for **five-to-six electricity-relevant near-surface variables**, judged
on **two** axes simultaneously:

1. **Distributional fidelity** against ERA5 (the observational target), and
2. **Climate-signal fidelity** against the driving GCM out to 2100.

Neither alone is sufficient: a method can match ERA5 distributions yet drift
away from the GCM's large-scale change signal (and vice-versa).

## 2. Units (normative)

Raw loader data is converted to physical units **at load time** and methods
operate on physical units. Units are **never** converted inside a method
function.

| CMIP6 short name | Quantity | Unit |
| --- | --- | --- |
| `tas` | near-surface air temperature | K |
| `pr` | precipitation (rate) | mm/day |
| `psl` | sea-level pressure | Pa |
| `sfcWind` | 10 m wind speed | m/s |
| `sfcWindmax` | daily-max 10 m wind speed | m/s |
| `rsds` | surface downwelling shortwave radiation | W/m² |

- Single source of truth: `src/dashboard/plot_meta.py::VAR_UNITS`.
- Every plot axis / colorbar that shows one of these MUST name its unit
  (`tas (K)`, `Δ tas (K)`, …). Unitless quantities (correlation r) MUST be
  labelled `(unitless)`.
- Change (Δ) maps MUST be prefixed with `Δ`.

## 3. Data invariants (normative)

- **Non-negativity** after correction for the physically non-negative variables
  `pr`, `sfcWind`, `sfcWindmax`, `rsds`: `np.clip(values, 0.0, None)`.
  (Code uses per-method `PHYS_MIN`/`CLIP_MIN` dicts; `evaluate.py` currently
  clips only `pr`/`sfcWind`/`sfcWindmax` — see `fix_plan.md`.)
- **Grid**: all regions are 57 (lat) × 65 (lon) at 0.25°; `lat` is stored
  north→south, `lon` west→east.
- **Time**: ERA5 covers 1979–2022 (France 1979–2020); calibration 1980–1999;
  evaluation 2000–2020 vs ERA5 and 2000–2100 vs GCM.
- **cftime**: after `bicubic_regrid`, GCM `time` carries `cftime`; it MUST be
  re-stamped to the ERA5 `time` coordinate before alignment/saving.
- **CDS area order**: `[North, West, South, East]`.

## 4. Acceptance criteria (what "correct" means)

A baseline result is accepted when all of the following hold:

- `results/{region}/{method}/metrics.json` contains, for every shipped variable:
  the `vs_obs` block, the `vs_gcm` block, and a `_delta` block with both
  `2040_2060` and `2080_2100` periods populated with finite numbers.
  (Verified benchmark-wide; see `fix_plan.md`.)
- `plots/` contains the per-variable `.npz` map data and `plot_data.json` exists.
- Predictions follow the standard format (`predictions.pt`, `variables.npy`,
  `eval_dates.npy`, `lat.npy`, `lon.npy`) via `save_predictions`.
- Physical invariants of §3 hold.

**Oracles / references** (agreement is within numerical tolerance, never
bit-for-bit): ERA5 = observational target; the driving GCM = climate-signal
reference; published metric definitions in `src/metrics/`.

## 5. Development protocol

This repository follows the legoESM / Ralph agent workflow:

- **One item per Ralph loop**, chosen from `fix_plan.md`.
- **Backpressure**: after every change, run the test for the unit touched
  (`python3 -m pytest <test> -q`). `tests/test_plot_meta.py` is the fast gate for
  dashboard plot metadata.
- **Adversarial review**: every substantial change is reviewed by an agent from
  a different model family than the one that wrote it.
- **No placeholders**: full implementations only.
- **Test the contract, not the implementation**: tests assert the clauses above,
  and each tripwire is itself non-vacuous (it iterates the real type/variable
  lists, so adding a new plot type without a caption fails).
- When the tests pass: update `fix_plan.md`, commit, push, and bump the patch tag.
