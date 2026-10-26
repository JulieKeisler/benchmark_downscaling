"""France-parity calendar covariates for arbitrary regions.

Replicates the column schema and value semantics of the French TAM dataset
(``dataset_30min_full_imputed.csv``), verified byte-for-byte against France rows:
  Date_1          = YYYYMMDD of the *UTC* day (timestamps are local civil time,
                    offset is subtracted first). France verified 100%: in CET
                    (UTC+1) local 00:00/00:30 -> previous day; in CEST (UTC+2)
                    local 00:00-01:30 -> previous day. The useful daily row
                    (tod==30, local 15:00) always maps to the same-day label.
  month, year     = calendar month / year.
  tod             = 30-min slot index of the *local civil* timestamp
                    (00:00 -> 0 ... 23:30 -> 47).
  toy             = round(day-of-year / days_in_year(year), 3)
                    (verified 100% on France: 366-day years use /366).
  week_number     = ISO week year * 100 + ISO week (2012-01-01 -> 201152).
  DayValidity     = 0 on DST transition dates and the following day
                    (France flags 4 dates/yr: spring/fall boundary + next day).
  day_type_week   = date.weekday()  (Mon=0 .. Sun=6; 100% match on France).
  day_type_jf     = 1 on public holidays.
  day_type_vjf    = 1 on the day before a public holiday.
  day_type_ljf    = 1 on the day after a public holiday.
  day_type_week_jf= 1 on weekends or public holidays (100% match on France).
  day_type_hc / dst_repair_target = 0 (unused in France TAM).
  period_hour_changed = 1 while daylight-saving time is in effect.
  period_holiday / zone_a/b/c = school-vacation flags (France-specific; AU/US
    have no equivalent calendar, so kept 0 for those regions).
  period_christmas = 1 from Dec 20 through Jan 6 (fixed window approximation
    of France's year-varying school Christmas break).
  period_summer   = AU: Dec-Feb ; US: Jun-Aug ; FR: Jun-Aug.
  day_type_week_period_hour_changed = day_type_week*10 + period_hour_changed
    (matches France, e.g. Sunday+Dst -> 61, Saturday+none -> 60).
  day_type_week_jf_period_holiday   = day_type_week_jf*10 + period_holiday
    (matches France, e.g. holiday weekend -> 11).

Time zone convention: timestamps are *local civil time* (no tz). DST
transitions per region are hard-coded by rule:
  france:     last Sun Mar -> last Sun Oct
  us_central: 2nd Sun Mar  -> 1st Sun Nov
  australia:  1st Sun Oct  -> 1st Sun Apr (NSW, VIC and SA observe it)
"""

from __future__ import annotations

import datetime as dt

import holidays as hl
import pandas as pd

CAL_COLS = [
    "date",
    "Date_1",
    "month",
    "year",
    "tod",
    "toy",
    "week_number",
    "DayValidity",
    "day_type_week",
    "day_type_jf",
    "day_type_vjf",
    "day_type_ljf",
    "day_type_week_jf",
    "day_type_hc",
    "dst_repair_target",
    "period_hour_changed",
    "period_holiday",
    "period_holiday_zone_a",
    "period_holiday_zone_b",
    "period_holiday_zone_c",
    "period_christmas",
    "period_summer",
    "day_type_week_period_hour_changed",
    "day_type_week_jf_period_holiday",
]


def _nth_weekday_of_month(year: int, month: int, weekday: int, nth: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=nth - 1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    first_next = dt.date(year, month + 1, 1) if month < 12 else dt.date(year + 1, 1, 1)
    d = first_next - dt.timedelta(days=1)
    while d.weekday() != weekday:
        d -= dt.timedelta(days=1)
    return d


def dst_boundaries(region: str, years) -> list[tuple[dt.date, dt.date]]:
    out = []
    for y in sorted(set(int(y) for y in years)):
        if region == "france":
            start, end = _last_weekday(y, 3, 6), _last_weekday(y, 10, 6)
        elif region == "us_central":
            start, end = _nth_weekday_of_month(y, 3, 6, 2), _nth_weekday_of_month(y, 11, 6, 1)
        elif region == "australia":
            start, end = _nth_weekday_of_month(y, 10, 6, 1), _nth_weekday_of_month(y + 1, 4, 6, 1)
        out.append((start, end))
    return out


def _is_dst(region: str, d: dt.date) -> bool:
    if region == "australia":
        years = [d.year - 1, d.year]
    else:
        years = [d.year]
    for s, e in dst_boundaries(region, years):
        if s <= d < e:
            return True
    return False


def _utc_offset_hours(region: str, t) -> int:
    if region == "us_central":
        base, dst_off = -6, -5
    elif region == "france":
        base, dst_off = 1, 2
    elif region == "australia":
        base, dst_off = 10, 11
    else:
        raise ValueError(f"unknown region {region!r}")
    return dst_off if _is_dst(region, t.date()) else base


def holiday_sets(region: str, years):
    years = sorted(set(int(y) for y in years))
    if region in ("france", "us_central"):
        h = hl.France(years=years) if region == "france" else hl.US(years=years)
        return set(h.keys())
    if region == "australia":
        return set().union(*(hl.AU(years=years, state=st).keys() for st in ("NSW", "VIC", "SA")))
    raise ValueError(f"unknown region {region!r}")


def build_calendar(times: pd.DatetimeIndex, region: str) -> pd.DataFrame:
    times = pd.DatetimeIndex(times).sort_values()
    dates = times.normalize()
    years = sorted(set(int(y) for y in times.year))
    holidays_union = holiday_sets(region, years)
    boundaries = dst_boundaries(region, years)
    unique_dates = dates.date
    dst_days: set[dt.date] = set(d for d in unique_dates if _is_dst(region, d))
    invalid_days: set[dt.date] = set()
    for s, e in boundaries:
        for d in (s, s + dt.timedelta(days=1), e, e + dt.timedelta(days=1)):
            invalid_days.add(d)

    n = len(times)
    year = [int(t.year) for t in times]
    month = [int(t.month) for t in times]
    weekday = [t.weekday() for t in times]

    date_1 = []
    for t in times:
        offset = _utc_offset_hours(region, t)
        utc = t - dt.timedelta(hours=offset)
        date_1.append(int(utc.strftime("%Y%m%d")))

    doy = [t.dayofyear for t in times]
    toy = [round(x / (366 if (t.year % 4 == 0 and (t.year % 100 != 0 or t.year % 400 == 0)) else 365), 3) for t, x in zip(times, doy)]
    iso_year = [t.isocalendar().year for t in times]
    iso_week = [t.isocalendar().week for t in times]
    week_number = [y * 100 + w for y, w in zip(iso_year, iso_week)]

    td = times.hour * 2 + (times.minute >= 30)

    jf = [int(d in holidays_union) for d in dates.date]
    vjf = [int((d + dt.timedelta(days=1)) in holidays_union) for d in dates.date]
    ljf = [int((d - dt.timedelta(days=1)) in holidays_union) for d in dates.date]
    weekend_or_jf = [1 if (w >= 5 or j) else 0 for w, j in zip(weekday, jf)]
    dst_flag = [int(d in dst_days) for d in dates.date]
    valid = [int(d not in invalid_days) for d in dates.date]

    christmas = [1 if (d.month == 12 and d.day >= 20) or (d.month == 1 and d.day <= 6) else 0 for d in dates.date]
    if region == "australia":
        summer = [1 if d.month in (12, 1, 2) else 0 for d in dates.date]
    else:
        summer = [1 if d.month in (6, 7, 8) else 0 for d in dates.date]

    holiday_flag = [int(d in holidays_union) for d in dates.date]
    period_holiday = holiday_flag if region == "france" else [0] * n

    out = pd.DataFrame(
        {
            "date": times.strftime("%Y-%m-%d %H:%M:%S"),
            "Date_1": [float(x) for x in date_1],
            "month": month,
            "year": year,
            "tod": td,
            "toy": toy,
            "week_number": week_number,
            "DayValidity": valid,
            "day_type_week": weekday,
            "day_type_jf": jf,
            "day_type_vjf": vjf,
            "day_type_ljf": ljf,
            "day_type_week_jf": weekend_or_jf,
            "day_type_hc": 0.0,
            "dst_repair_target": 0.0,
            "period_hour_changed": dst_flag,
            "period_holiday": period_holiday,
            "period_holiday_zone_a": [0] * n,
            "period_holiday_zone_b": [0] * n,
            "period_holiday_zone_c": [0] * n,
            "period_christmas": christmas,
            "period_summer": summer,
            "day_type_week_period_hour_changed": [w * 10 + p for w, p in zip(weekday, dst_flag)],
            "day_type_week_jf_period_holiday": [w * 10 + p for w, p in zip(weekend_or_jf, period_holiday)],
        }
    )
    return out[CAL_COLS]
