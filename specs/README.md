# Specifications

One spec per component, written as requirements (not as a description of the
code). Ralph re-reads these every loop; they are the "stack" that is
deterministically re-allocated so the agent never has to guess intent.

Rules:

- A spec states **what must be true**, with acceptance criteria a test can check.
- Keep each spec short and current. Delete stale statements rather than letting
  them rot.
- Where a spec and `CONTRACT.md` overlap, `CONTRACT.md` wins (it is normative for
  units, invariants and acceptance).

Specs:

| File | Component |
| --- | --- |
| `specs/dashboard_plots.md` | Dashboard plot captions, units, tabs |
| `specs/metrics.md` | Metric definitions and where they are computed |
| `specs/data_protocol.md` | Regions, variables, periods, alignment, downloads |

Known gaps (do not assume implemented — search first):

- Dashboard statistics table / radar / TAM tabs are not yet fully specified.
- Sea-ice / biogeochemistry are out of scope for this benchmark.
