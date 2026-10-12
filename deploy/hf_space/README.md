---
title: Downscaling Benchmark
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# Downscaling Benchmark for the Electricity System

Interactive dashboard comparing downscaling / bias-correction methods for the
electricity system (ERA5 reference, CMIP6 GCMs, France / Australia / US-central).

- **Code:** https://github.com/JulieKeisler/benchmark_downscaling
- The app **clones the code from GitHub** at startup and **fetches the results
  data from a Hugging Face dataset repo** (see `app.py`).

## Space variables (optional)

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_REPO_URL` | `https://github.com/JulieKeisler/benchmark_downscaling` | Code source |
| `BENCH_REPO_REF` | `main` | Git ref/branch to check out |
| `HF_RESULTS_REPO` | `JulieKeisler/benchmark-downscaling-results` | Dataset repo holding `results.tar.gz` |
| `RESULTS_ROOT` | `/data/results` | Where the data archive is extracted |

Deploy/update the data and code with `tools/deploy_hf.py`.
