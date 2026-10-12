"""Temporal climatology / anomaly decomposition (LF_t / HF_t in the paper's notation)."""

import torch
import xarray as xr
from tqdm import tqdm

from .data_utils import stack_variables_as_channels
from .time_utils import add_noleap_doy, remove_29_feb


def climatology(datas, delta: int = 20, mean_mode: str = "local", seasonal: bool = True):
    """Rolling K-year (delta) day-of-year climatology and anomaly.

    seasonal=True (used everywhere in this project): for each year y, the
    climatology is the mean over the *preceding* `delta` years at the same
    day-of-year (Eq. "climatology" in the paper), so the first `delta` years
    of each input series are consumed as history and produce no output.
    """
    results, clim_results = [], []
    window = delta * 365

    if seasonal:
        for data in tqdm(datas, desc="Datasets"):
            data = add_noleap_doy(remove_29_feb(data))
            start_year = int(data.time.dt.year.min())
            end_year = int(data.time.dt.year.max())

            yearly_outputs, yearly_clim = [], []
            for year in range(start_year + delta, end_year + 1):
                past = data.sel(time=slice(f"{year - delta}-01-01", f"{year - 1}-12-31"))
                current = data.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

                clim_mean = past.groupby("dayofyear_noleap").mean("time")
                if mean_mode == "global":
                    clim_mean = clim_mean.mean(dim=["latitude", "longitude"])

                anomaly = current.groupby("dayofyear_noleap") - clim_mean
                clim_mean_aligned = clim_mean.sel(dayofyear_noleap=current.dayofyear_noleap).assign_coords(
                    time=current.time
                )
                yearly_outputs.append(anomaly)
                yearly_clim.append(clim_mean_aligned)

            results.append(xr.concat(yearly_outputs, dim="time"))
            clim_results.append(xr.concat(yearly_clim, dim="time"))
        return results, clim_results

    for data in tqdm(datas, desc="Datasets"):
        data = add_noleap_doy(remove_29_feb(data))
        if mean_mode == "global":
            spatial_mean = data.mean(dim=["latitude", "longitude"])
            clim_mean = spatial_mean.shift(time=1).rolling(time=window, min_periods=window, center=False).mean()
            anomaly = spatial_mean - clim_mean
        else:
            try:
                import dask.array  # noqa: F401
                lat_chunks = max(1, data.sizes.get("latitude", 1) // 4)
                lon_chunks = max(1, data.sizes.get("longitude", 1) // 4)
                data_chunked = data.chunk({"time": -1, "latitude": lat_chunks, "longitude": lon_chunks})
            except ImportError:
                data_chunked = data
            clim_mean = data_chunked.shift(time=1).rolling(time=window, min_periods=window, center=False).mean()
            if clim_mean.chunks:
                clim_mean = clim_mean.compute()
            anomaly = data - clim_mean

        valid_mask = clim_mean.time[window:]
        clim_mean = clim_mean.sel(time=valid_mask)
        anomaly = anomaly.sel(time=valid_mask)
        results.append(anomaly)
        clim_results.append(clim_mean)

    return results, clim_results


def apply_climatology_and_save(datas, vars_name, save_path=None, clim_save_path=None,
                                delta: int = 20, mean_mode: str = "local", save: bool = True,
                                seasonal: bool = True):
    results, clim_results = climatology(datas, delta, mean_mode, seasonal)
    results = xr.align(*results, join="inner")
    clim_results = xr.align(*clim_results, join="inner")

    def to_tensor(aligned_results):
        assert len(vars_name) == len(set(vars_name)), f"duplicate variable names in vars_name: {vars_name}"
        named = {vars_name[i]: res[vars_name[i]] for i, res in enumerate(aligned_results)}
        return stack_variables_as_channels(named)

    final_tensor = to_tensor(results)
    clim_tensor = to_tensor(clim_results)

    if save:
        torch.save(final_tensor, save_path)
        torch.save(clim_tensor, clim_save_path)
    return results, clim_results
