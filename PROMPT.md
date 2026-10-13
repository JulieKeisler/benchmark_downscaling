# Ralph loop prompt

You are one iteration of a persistent coding-agent loop working in this repo
(the Downscaling Benchmark for the Electricity System). You are launched fresh
every loop, so everything you need must come from the files below.

## Read first (the stack — re-allocated every loop)

- `CONTRACT.md` — normative units, invariants, acceptance criteria.
- `specs/*.md` — per-component requirements.
- `fix_plan.md` — prioritized TODO. **It may be out of date; fix it.**

## Task

1. Read `fix_plan.md` and pick **the single most important open item**. Prefer
   item 1 unless a human has redirected. State clearly which item you are doing.
2. Before changing anything, **search the codebase** (don't assume a thing is not
   already implemented). Use parallel subagents for searching; use **only one
   subagent** for any build/test run to avoid backpressure.
3. Implement the item **fully** — no placeholders, no minimal stubs. If a helper
   is missing, add it. Respect `CONTRACT.md` (units are not converted inside
   method functions; non-negativity clipping; cftime re-stamp after regrid).
4. **Backpressure**: after the change, run the test for the unit you touched.
   - Dashboard plot metadata: `python3 -m pytest tests/test_plot_meta.py -q`
   - Full suite: `python3 -m pytest tests/test_metrics.py tests/test_data_processing.py -q`
   - Any dashboard edit: also `python3 -m py_compile src/dashboard/dashboard.py`
     and load it headlessly (AppTest) to catch runtime errors.
   If the change has no test, **write one** and, in the test, capture **why it
   matters** — future loops have no memory of this reasoning.
5. If a test unrelated to your work fails, resolve it as part of this increment.
6. Update `fix_plan.md`: mark the item done, record findings, prune stale items.
7. When the tests pass: `git add -A`, commit with a message describing the
   change, `git push`, and bump the patch tag (start at `0.0.1` if no tag
   exists). Do not tag if tests fail.

## Signs (do not ignore)

- Do NOT implement placeholder or simple implementations. Full implementations.
- Keep `AGENTS.md` brief and current with build/test learnings (it is local-only,
  gitignored — that is fine).
- For any bug you notice, fix it or record it in `fix_plan.md` (documented for a
  later loop) even if it is unrelated to the current item.
- Deterministically re-read the same stack each loop; spend as little context as
  possible — delegate summarising large outputs to subagents.
