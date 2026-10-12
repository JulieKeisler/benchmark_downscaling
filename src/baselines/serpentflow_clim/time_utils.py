"""Calendar utilities: fixed 365-day (no-leap) calendar used throughout the project."""

import numpy as np
import xarray as xr
import torch

_CUM_DAYS_NOLEAP = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def date_str_to_doy_noleap(date_str: str) -> int:
    """'YYYY-MM-DD' -> day-of-year in [1, 365], ignoring leap years."""
    y, m, d = (int(p) for p in date_str.split("-"))
    assert 1 <= m <= 12, f"bad month in {date_str}"
    assert 1 <= d <= 31, f"bad day in {date_str}"
    doy = _CUM_DAYS_NOLEAP[m - 1] + d
    assert 1 <= doy <= 365, f"computed doy {doy} out of range for {date_str}"
    return doy


def build_doy_array(start_date: str, n: int, num_doy: int = 365) -> torch.Tensor:
    """DOY (1..num_doy) for n consecutive daily samples starting at `start_date`.

    Assumes the n samples are truly consecutive calendar days with no gaps or
    duplicates (leap days already removed). This is not checked here -- see
    `data_utils.stack_variables_as_channels` for the alignment guard that
    should run before tensors reach this point.
    """
    start = date_str_to_doy_noleap(start_date)
    idx = (torch.arange(n, dtype=torch.long) + (start - 1)) % num_doy
    return (idx + 1).long()


def remove_29_feb(data):
    return data.sel(time=~((data["time.month"] == 2) & (data["time.day"] == 29)))


def add_noleap_doy(data):
    doy = data.time.dt.dayofyear
    is_leap = data.time.dt.is_leap_year
    doy_fixed = xr.where((is_leap) & (doy > 59), doy - 1, doy)
    return data.assign_coords(dayofyear_noleap=doy_fixed)


def align_time_to_midnight(ds):
    """Normalize the time coordinate to 00:00:00 (drops sub-day offsets)."""
    ds = ds.copy()
    ds["time"] = ds.indexes["time"].normalize()
    return ds


def assert_no_leap_no_gaps(data, time_dim: str = "time") -> None:
    """Fail loudly if the (leap-day-removed) time axis isn't exactly consecutive.

    Was missing from the original pipeline: variables were converted to
    tensors and combined with `torch.cat(..., dim=1)` with no check that
    their time indices actually lined up day-for-day. A single missing or
    duplicated date in any one variable's source file would silently
    misalign every downstream training/inference sample across channels.
    """
    times = data[time_dim]
    if (times.dt.is_leap_year & (times.dt.month == 2) & (times.dt.day == 29)).values.any():
        raise ValueError("time axis still contains a Feb-29: call remove_29_feb first")

    dt_days = (times.values[1:] - times.values[:-1]).astype("timedelta64[D]").astype(int)
    prev_month, prev_day, prev_leap = times.dt.month.values[:-1], times.dt.day.values[:-1], times.dt.is_leap_year.values[:-1]

    # After removing Feb 29, consecutive calendar days are 1 apart, except
    # for the single legitimate 2-day jump from Feb 28 to Mar 1 in a leap year.
    is_leap_feb28_to_mar1 = (dt_days == 2) & (prev_month == 2) & (prev_day == 28) & prev_leap
    bad = np.where((dt_days != 1) & ~is_leap_feb28_to_mar1)[0]
    if len(bad):
        examples = [(int(i), str(times.values[i]), str(times.values[i + 1])) for i in bad[:3]]
        raise ValueError(f"non-consecutive / duplicated dates found at {len(bad)} location(s), e.g. {examples}")


def assert_time_axes_aligned(named_arrays: "dict[str, xr.DataArray]", time_dim: str = "time") -> None:
    """Fail loudly unless every array shares the exact same, gap-free time axis.

    Combines the single-series check (`assert_no_leap_no_gaps`) with a
    cross-array equality check, so callers that need to combine several
    per-variable series (e.g. `data_utils.stack_variables_as_channels`,
    or `denormalize`'s `data`/`ref` pair) verify "these time axes agree" in
    one place instead of each re-deriving part of the check.
    """
    names = list(named_arrays.keys())
    ref_time = named_arrays[names[0]][time_dim]
    assert_no_leap_no_gaps(named_arrays[names[0]])
    for name in names[1:]:
        assert_no_leap_no_gaps(named_arrays[name])
        other_time = named_arrays[name][time_dim]
        if len(other_time) != len(ref_time) or not np.array_equal(other_time.values, ref_time.values):
            raise ValueError(
                f"time axis mismatch: '{names[0]}' has {len(ref_time)} steps "
                f"({ref_time.values[0]}..{ref_time.values[-1]}), '{name}' has {len(other_time)} steps "
                f"({other_time.values[0] if len(other_time) else 'n/a'}.."
                f"{other_time.values[-1] if len(other_time) else 'n/a'})"
            )
