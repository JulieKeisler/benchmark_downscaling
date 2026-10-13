# Spec — Data and protocol

## Regions (all 57 lat × 65 lon at 0.25°)

| Region | Box `[N, W, S, E]` | Native GCM |
| --- | --- | --- |
| `france` | `[55.0, -6.0, 41.0, 10.0]` | `ipsl_cm6a_lr` |
| `australia` | `[-24.0, 136.0, -38.0, 152.0]` | `cnrm_cm6_1` |
| `us_central` | `[48.0, -105.0, 34.0, -89.0]` | `cesm2` |

CDS area order is `[North, West, South, East]` (NOT W, S, E, N).

## Variables and units

See `CONTRACT.md §Units`. All six are shipped except where the source lacks them:
`us_central` has no `sfcWindmax` (CESM2 provides none); known gaps are documented
in `src/data_download/README.md` and must NOT be "fixed" by inventing data.

## Periods

- ERA5 reference: 1979–2022 (France 1979–2020).
- Calibration: 1980–1999.
- Evaluation: 2000–2020 vs ERA5; 2000–2100 vs GCM (SSP scenarios).

## Alignment

GCM and ERA5 must be time-aligned (inner join on evaluation window). After
`bicubic_regrid`, re-stamp GCM `time` to the ERA5 coordinate (cftime gotcha,
`CONTRACT.md §3`).

## Downloads

- ERA5 reference = WB2 archive crop on Jean-Zay; CDS scripts are fallback only.
- GCMs = `esgf/download_cmip6_esgf.py` (multi-scenario; dkrz esgf3 mirror).

## Acceptance

Every (region, method) coupling exposes the standard predictions format and a
complete `metrics.json` (see `CONTRACT.md §4`).
