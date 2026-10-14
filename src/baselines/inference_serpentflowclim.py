"""SerpentFlow-CLIM — climatology-aware flow-matching downscaling (Haidar 2026).

Faithful port of the reference `serpentflow_clim` package (shipped unchanged
under src/baselines/serpentflow_clim/) into the benchmark's 6-variable,
per-region evaluation setup. Three phases, runnable standalone:

    python src/baselines/inference_serpentflowclim.py \
        --phase prep --region france \
        --era5-dir data/era5 --gcm-dir data/cmip6/france/ipsl_cm6a_lr \
        --scenario ssp370 --static data/static/france/static_fields_v1.pt \
        --pred-dir results/serpentflow-clim/france
    python src/baselines/inference_serpentflowclim.py --phase train --region france --pred-dir ...
    python src/baselines/inference_serpentflowclim.py --phase infer --region france --pred-dir ...

Pipeline (all stages identical to the reference implementation):

  prep    LF_s (fourier spatial low-pass, CF=1) of ERA5+GCM; 20-year seasonal
          climatology (LF_t); CDF-t v3 correction of the GCM climatology
          (calibrated on 1980-1999, applied to the eval window). Training
          tensors 1980-1999: tas decomposed (source=HF_t(LF_s), target=x-clim),
          other vars raw (source=LF_s, target=x). Inference conditioning
          2000-eval_end: tas=HF_t(LF_s(GCM)); others=LF_s - LF_t + CDFt---
          See ablation_02/jz_infer_ablation for the channel recipe.
  train   UNetFlow(C_in=6, C_out=6, static_dim=4, num_doy=365, ERA5 cfg),
          400 epochs, batch 64, lr 2e-4, EMA decay 0.999.
  infer   8-member dopri5 ODE ensemble over the conditioning tensor;
          reconstruct physical units with the conditioning tensor's channel
          z-score stats (exact global inverse of the normalization the dataset
          applies at inference -- not the as-submitted self-referential
          denormalize), reinject the CDF-t-corrected GCM climatology for tas,
          clip non-negative pr/sfcWind/sfcWindmax/rsds, save_predictions.

Physical units throughout: tas K, pr mm/day, sfcWind*/sfcWind m/s, psl Pa,
rsds W m-2 (loaders already convert raw-to-physical; see AGENTS.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "evaluation"))

from data_processing import (  # noqa: E402
    DEFAULT_VARIABLES, bicubic_regrid, compute_norm_stats, load_era5, load_gcm,
    save_predictions,
)

from serpentflow_clim import config, model  # noqa: E402
from serpentflow_clim.cdft import run_cdft_20y  # noqa: E402
from serpentflow_clim.climatology import climatology  # noqa: E402
from serpentflow_clim.data_utils import (  # noqa: E402
    load_static_conditioning, stack_variables_as_channels,
)
from serpentflow_clim.dataset import SerpentFlowDataset  # noqa: E402
from serpentflow_clim.ema import EMA  # noqa: E402
from serpentflow_clim.filters import fourier_tensor_batch  # noqa: E402
from serpentflow_clim.inference import odeint_torchdiffeq_adaptive  # noqa: E402
from serpentflow_clim.time_utils import (  # noqa: E402
    align_time_to_midnight, remove_29_feb,
)

CF = 1.0                      # spatial low-pass cutoff (r_cut)
TRAIN_SLICE = slice("1980-01-01", "1999-12-31")
TRAIN_START = "1980-01-01"
NOISE_SCALE_TRAIN = (1.0,) * 7                   # placeholder; resized to n_ch
NOISE_SCALE_INFER = (0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)   # tas λ=0.4, others 1.0
N_ENSEMBLE = 8
LAT_LON = ("latitude", "longitude")


class SFT:
    """SerpentFlow-CLIM tensor prep / train / infer for one region."""

    def __init__(self, region: str, era5_dir: Path, gcm_dir: Path, scenario: str,
                 static_path: Path, pred_dir: Path, eval_start: int, eval_end: int,
                 variables: list[str] | None = None):
        self.region = region
        self.era5_dir = Path(era5_dir)
        self.gcm_dir = Path(gcm_dir)
        self.scenario = scenario
        self.static_path = Path(static_path)
        self.pred_dir = Path(pred_dir)
        self.prep_dir = self.pred_dir / "prep"
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.variables = list(variables or DEFAULT_VARIABLES)
        if self.variables[0] != "tas":
            raise ValueError(
                f"SerpentFlow-CLIM decomposes channel 0 as tas; got "
                f"variables[0]={self.variables[0]!r}. Keep tas first.")
        self.prep_dir.mkdir(parents=True, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _clean(self, da: xr.DataArray) -> xr.DataArray:
        from serpentflow_clim.time_utils import remove_29_feb
        da = remove_29_feb(da)
        if "cftime" in str(type(da.indexes["time"])):
            # GCM native calendar (daily, e.g. 12:00): normalize to datetime64 midnight
            new = np.array([np.datetime64(str(t)[:10]) for t in da.time.values],
                           dtype="datetime64[ns]")
            da = da.assign_coords(time=new)
        else:
            da = align_time_to_midnight(da)
        return da.rename({"lat": "latitude", "lon": "longitude"})

    def _lfs_ds(self, da: xr.DataArray) -> xr.Dataset:
        """LF_s (spatial low-pass) of a DataArray -> Dataset[var]."""
        pt = torch.from_numpy(da.to_numpy()).unsqueeze(1)
        out = fourier_tensor_batch(pt, CF)
        return xr.Dataset({da.name: xr.DataArray(
            out.squeeze(1).numpy(), dims=("time",) + LAT_LON,
            coords={"time": da.time, "latitude": da[LAT_LON[0]], "longitude": da[LAT_LON[1]]})})

    # ── prep ─────────────────────────────────────────────────────────────────

    def prep(self) -> None:
        t0 = time.time()

        # ---- ERA5 train (1960-1999) + GCM (1960-eval_end) ----
        era5 = {}
        for var in self.variables:
            era5[var] = self._clean(load_era5(self.era5_dir, var, 1960, 1999))
        print(f"ERA5 loaded (1960-1999) in {time.time()-t0:.1f}s", flush=True)

        gcm = {}
        for var in self.variables:
            g = load_gcm(self.gcm_dir, var, 1960, self.eval_end, self.scenario)
            gcm[var] = self._clean(bicubic_regrid(g, era5[var].latitude.values,
                                                  era5[var].longitude.values))
        print(f"GCM loaded + bicubic-regridded ({1960}-{self.eval_end}) in "
              f"{time.time()-t0:.1f}s", flush=True)

        # ---- LF_s + climatology (LF_t) for every variable ----
        era5_lfs = {v: self._lfs_ds(era5[v]) for v in self.variables}
        gcm_lfs = {v: self._lfs_ds(gcm[v]) for v in self.variables}

        _, era5_clim = climatology(list(era5_lfs.values()), delta=20,
                                   mean_mode="local", seasonal=True)
        _, gcm_clim = climatology(list(gcm_lfs.values()), delta=20,
                                  mean_mode="local", seasonal=True)
        era5_clim = {v: c for v, c in zip(self.variables, era5_clim)}
        gcm_clim = {v: c for v, c in zip(self.variables, gcm_clim)}
        print(f"climatology computed in {time.time()-t0:.1f}s", flush=True)

        # ---- CDF-t v3 correction of the GCM climatology over the eval window ----
        gcm_test_slice = slice(f"{self.eval_start}-01-01", f"{self.eval_end}-12-31")
        cdft3 = {}
        n_days = None
        for var in self.variables:
            e_train = era5_clim[var].sel(time=TRAIN_SLICE)
            g_train = gcm_clim[var].sel(time=TRAIN_SLICE)
            g_test = gcm_clim[var].sel(time=gcm_test_slice)
            _, _, c3 = run_cdft_20y(gcm_test_clim=g_test, era5_train_clim=e_train,
                                    gcm_train_clim=g_train, var=var, required=[3])
            cdft3[var] = c3.transpose("time", *LAT_LON)
            n_days = len(cdft3[var]["time"])
        print(f"CDF-t v3 climatology correction in {time.time()-t0:.1f}s "
              f"(eval window {n_days} days)", flush=True)

        # ---- training tensors: tas decomposed, others raw ----
        v0 = self.variables[0]                      # tas
        tas_anom_source = (era5_lfs[v0][v0].sel(time=TRAIN_SLICE)
                           - era5_clim[v0][v0].sel(time=TRAIN_SLICE))
        tas_anom_target = (era5[v0].sel(time=TRAIN_SLICE)
                           - era5_clim[v0][v0].sel(time=TRAIN_SLICE))

        train_source = {v: (tas_anom_source if v == v0
                            else era5_lfs[v][v].sel(time=TRAIN_SLICE)) for v in self.variables}
        train_target = {v: (tas_anom_target if v == v0
                            else era5[v].sel(time=TRAIN_SLICE)) for v in self.variables}

        src = stack_variables_as_channels(train_source)
        tgt = stack_variables_as_channels(train_target)
        torch.save(src, self.prep_dir / "train_source.pt")
        torch.save(tgt, self.prep_dir / "train_target.pt")
        print(f"train tensors: src {tuple(src.shape)} tgt {tuple(tgt.shape)} "
              f"({self.variables[0]} decomposed, others raw)", flush=True)

        # ---- inference conditioning tensor (clim5_style) ----
        cond = {}
        for v in self.variables:
            lfs_v = gcm_lfs[v][v].sel(time=gcm_test_slice)
            clim_v = gcm_clim[v][v].sel(time=gcm_test_slice)
            if v == v0:                                    # tas: HF_t(LF_s)
                cond[v] = lfs_v - clim_v
            else:                                          # LF_s - LF_t + CDFt(LF_t)
                cond[v] = lfs_v - clim_v + cdft3[v].sel(time=gcm_test_slice)
        cond_t = stack_variables_as_channels(cond)
        torch.save(cond_t, self.prep_dir / "inference_cond.pt")
        print(f"inference conditioning: {tuple(cond_t.shape)}", flush=True)

        # ---- tas climatology reinjection term (CDF-t-corrected), eval window ----
        clim_tas = cdft3[v0].sel(time=gcm_test_slice)
        torch.save(torch.from_numpy(clim_tas.to_numpy()).float(),
                   self.prep_dir / "clim_tas_cdft3.pt")
        print(f"tas climatology reinjection term: {tuple(clim_tas.shape)}", flush=True)

        # ---- reconstruction reference: channel z-score stats of the CONDITIONING
        # tensor (input=target=cond at inference, so exact_global_inverse undoes
        # the very normalization SerpentFlowDataset applies on load) ----
        cond_stats = compute_norm_stats(cond_t, self.variables)
        (self.prep_dir / "cond_stats.json").write_text(json.dumps(cond_stats, indent=2))
        print("conditioning channel stats (for reconstruction): "
              f"{ {v: (round(s['mean'],4), round(s['std'],4)) for v, s in cond_stats.items()} }",
              flush=True)

        # ---- static conditioning ----
        torch.save(load_static_conditioning(str(self.static_path), standardize=False),
                   self.prep_dir / "static.pt")

        # ---- eval/grid metadata ----
        ref_times = [str(t)[:10] for t in cdft3[self.variables[0]]["time"].values]
        np.save(self.prep_dir / "eval_dates.npy", np.array(ref_times))
        np.save(self.prep_dir / "lat.npy", era5[self.variables[0]].latitude.values)
        np.save(self.prep_dir / "lon.npy", era5[self.variables[0]].longitude.values)

        print(f"PREP complete in {time.time()-t0:.1f}s -> {self.prep_dir}", flush=True)

    # ── train ────────────────────────────────────────────────────────────────

    def train(self, device: str, resume: bool = True) -> None:
        t0 = time.time()
        n_ch = len(self.variables)
        dataset = SerpentFlowDataset(
            str(self.prep_dir / "train_source.pt"),
            str(self.prep_dir / "train_target.pt"),
            start_date=TRAIN_START, apply_noise=True, num_doy=365,
            noise_scale=NOISE_SCALE_TRAIN[:n_ch],
        )
        print(f"Dataset loaded: {len(dataset)} samples in {time.time()-t0:.1f}s", flush=True)

        m = model.UNetFlow(C_in=n_ch, C_out=n_ch, static_dim=4, num_doy=365,
                           **config.make_model_cfg("ERA5"))
        static = load_static_conditioning(str(self.prep_dir / "static.pt"),
                                          standardize=False)
        m.set_static(static)
        print(f"UNetFlow(C_in={n_ch}) {sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.2f}M "
              f"params, static {tuple(static.shape)}", flush=True)

        ema_model = EMA(model=m)
        ckpt_path = self.pred_dir / "checkpoints" / "fm_serpentflowclim_best.pth"
        checkpoint = None
        if resume and ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ema_model.load_state_dict(checkpoint["ema"])
            print(f"Resuming from {ckpt_path}", flush=True)

        tcfg = config.make_train_cfg("ERA5")   # batch 64, 400 epochs, lr 2e-4
        train_flow_matching = _import_train()
        train_flow_matching(
            model=ema_model, dataset=dataset, device=device, name="serpentflowclim",
            ckpt=checkpoint, mask=None, save_dir=str(self.pred_dir / "checkpoints"),
            **tcfg,
        )
        print(f"TRAIN complete in {time.time()-t0:.1f}s", flush=True)

    # ── infer ────────────────────────────────────────────────────────────────

    def infer(self, device: str, n_ensemble: int = N_ENSEMBLE) -> None:
        t0 = time.time()
        stats = json.loads((self.prep_dir / "cond_stats.json").read_text())
        n_ch = len(self.variables)

        m = model.UNetFlow(C_in=n_ch, C_out=n_ch, static_dim=4, num_doy=365,
                           **config.make_model_cfg("ERA5"))
        static = load_static_conditioning(str(self.prep_dir / "static.pt"),
                                          standardize=False)
        m.set_static(static)
        ema_model = EMA(model=m)
        ckpt = torch.load(self.pred_dir / "checkpoints" / "fm_serpentflowclim_best.pth",
                          map_location="cpu", weights_only=False)
        ema_model.load_state_dict(ckpt["ema"])
        inner = ema_model.model.to(device).eval()
        print(f"Loaded checkpoint {ckpt['epoch']} (loss {float(ckpt['loss']):.6f})", flush=True)

        cond = torch.load(self.prep_dir / "inference_cond.pt", map_location="cpu")
        n_days = cond.shape[0]
        t_span_days = f"{self.eval_start}-01-01"
        members = []
        for e in range(1, n_ensemble + 1):
            out_path = self.prep_dir / f"inference_ensemble{e}.pt"
            if out_path.exists():
                try:
                    exist = torch.load(out_path, map_location="cpu", weights_only=False)
                    assert exist.shape == cond.shape
                    members.append(exist)
                    print(f"  member {e}/{n_ensemble}: cached {tuple(exist.shape)}, skip", flush=True)
                    continue
                except Exception as exc:
                    print(f"  member {e}: cache invalid ({exc}), redo", flush=True)
            torch.manual_seed(1000 + e)
            dataset = SerpentFlowDataset(
                str(self.prep_dir / "inference_cond.pt"),
                str(self.prep_dir / "inference_cond.pt"),
                start_date=t_span_days, apply_noise=True, num_doy=365,
                noise_scale=NOISE_SCALE_INFER[:len(self.variables)],
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False,
                                                 num_workers=2)
            chunks = []
            te = time.time()
            with torch.no_grad():
                for data in loader:
                    x_state = data["noisy"].to(device)
                    st = data.get("stats").to(device) if data.get("stats") is not None else None
                    doy = data.get("doy").to(device) if data.get("doy") is not None else None
                    sample = odeint_torchdiffeq_adaptive(inner, x_state, stats=st, doy=doy,
                                                         device=device, method="dopri5").cpu()
                    chunks.append(sample)
            member = torch.cat(chunks, dim=0)
            torch.save(member, out_path)
            members.append(member)
            print(f"  member {e}/{n_ensemble}: {tuple(member.shape)} in "
                  f"{time.time()-te:.1f}s -> {out_path.name}", flush=True)

        mean_sample = torch.stack(members, dim=0).mean(dim=0)   # [T,C,H,W] normalized
        recon = self._reconstruct(mean_sample, stats)
        self._save(recon)
        print(f"INFER complete in {time.time()-t0:.1f}s -> {self.pred_dir}", flush=True)

    def save_member(self, member: int, out_dir: Path) -> None:
        """Write one cached ensemble member (not the mean) as standard predictions.

        Averaging members deflates the variance of a stochastic generator, so the
        benchmark scores single members; the cache comes from the infer phase.
        """
        stats = json.loads((self.prep_dir / "cond_stats.json").read_text())
        z = torch.load(self.prep_dir / f"inference_ensemble{member}.pt",
                       map_location="cpu", weights_only=False)
        self._save(self._reconstruct(z, stats), out_dir)
        print(f"member {member} -> {out_dir}", flush=True)

    # ── reconstruction & save ───────────────────────────────────────────────

    def _reconstruct(self, z: torch.Tensor, stats: dict) -> torch.Tensor:
        """Physical units from the normalized ODE output (exact global inverse
        of the conditioning-tensor channel z-score -- the normalization the
        dataset applies at inference, input=target=cond -- plus the
        CDF-t-corrected GCM climatology reinjection for tas)."""
        out = torch.zeros_like(z)
        v0 = self.variables[0]
        for c, var in enumerate(self.variables):
            mu, sd = float(stats[var]["mean"]), float(stats[var]["std"])
            out[:, c] = z[:, c] * sd + mu
        # tas: reinject the CDF-t-corrected GCM climatology (training target was
        # the anomaly x - LF_t(LF_s(x)); the conditioning removed the raw LF_t, so
        # reconstruction = anomaly + corrected climatology, per ablation_03 recipe).
        clim_tas = torch.load(self.prep_dir / "clim_tas_cdft3.pt", map_location="cpu")
        out[:, 0] = out[:, 0] + clim_tas
        # clip non-negative physical variables
        for c, var in enumerate(self.variables):
            if var in ("pr", "sfcWind", "sfcWindmax", "rsds"):
                out[:, c] = torch.clamp(out[:, c], min=0.0)
        return out

    def _save(self, tensor: torch.Tensor, out_dir: Path | None = None) -> None:
        lat = np.load(self.prep_dir / "lat.npy")
        lon = np.load(self.prep_dir / "lon.npy")
        dates = list(np.load(self.prep_dir / "eval_dates.npy", allow_pickle=True))
        ref = xr.Dataset(
            {self.variables[0]: (("time", "lat", "lon"), tensor[:, 0].numpy())},
            coords={"time": np.array(dates, dtype="datetime64[ns]"),
                    "lat": lat, "lon": lon},
        )
        save_predictions(tensor, self.variables, out_dir or self.pred_dir,
                         ref[self.variables[0]])


def _import_train():
    from serpentflow_clim.training import train_flow_matching  # noqa: PLC0415
    return train_flow_matching


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--phase", required=True, choices=["prep", "train", "infer", "member"])
    p.add_argument("--region", required=True)
    p.add_argument("--era5-dir", default=None)
    p.add_argument("--gcm-dir", default=None)
    p.add_argument("--scenario", default=None)
    p.add_argument("--static", default=None)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--eval-start", type=int, default=2000)
    p.add_argument("--eval-end", type=int, default=2020)
    p.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ensemble", type=int, default=N_ENSEMBLE)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--member", type=int, default=1,
                   help="--phase member: which cached ensemble member to write")
    p.add_argument("--out-dir", default=None,
                   help="--phase member: destination (default: --pred-dir)")
    args = p.parse_args()

    # Resolve region defaults from regions.yaml
    import yaml  # noqa: PLC0415
    regions = yaml.safe_load((_HERE.parent / "configs" / "data" / "regions.yaml").read_text())["regions"]
    if args.region not in regions:
        raise SystemExit(f"unknown region '{args.region}'; known: {sorted(regions)}")
    r = regions[args.region]
    era5_dir = Path(args.era5_dir or r["era5_dir"])
    gcm_dir = Path(args.gcm_dir or (Path(r["gcm_dir"]) / r["gcm"]))
    scenario = args.scenario or r["scenario"]
    static = Path(args.static or f"data/static/{args.region}/static_fields_v1.pt")

    sf = SFT(args.region, era5_dir, gcm_dir, scenario, static,
             Path(args.pred_dir), args.eval_start, args.eval_end, args.variables)
    if args.phase == "prep":
        sf.prep()
    elif args.phase == "train":
        sf.train(args.device, resume=not args.no_resume)
    elif args.phase == "infer":
        sf.infer(args.device, n_ensemble=args.ensemble)
    elif args.phase == "member":
        sf.save_member(args.member, Path(args.out_dir or args.pred_dir))


if __name__ == "__main__":
    main()