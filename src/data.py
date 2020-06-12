# ==============================================================================
# Data loading and processing functions
# ==============================================================================

import numpy as np
import pandas as pd

GAMMA = 1/5  # inverse infectious period.
STD = 3.5  # Gaussian smoothing std dev.


def _usa():
    """Returns a data frame of cumulative daily infections per state."""
    df = pd.read_csv('../data/usa.csv', usecols=['date', 'state', 'positive'],
                     parse_dates=['date'])
    df = df.rename(columns=dict(state='region', positive='cumulative'))
    return df.dropna(subset=['cumulative'])


def _za():
    """Returns a data frame of cumulative daily infections per province."""
    df = pd.read_csv('../data/za.csv', parse_dates=['YYYYMMDD'])
    df = df.drop(columns=['date', 'UNKNOWN', 'total', 'source'])
    df = df.rename(columns=dict(YYYYMMDD='date'))
    df = df.melt('date', var_name='region', value_name='cumulative')
    return df.dropna(subset=['cumulative'])


def _regions(df, top=None, min_positive=100):
    """Returns regions with the most cumulative infections.

    Args:
        top: if not `None`, only this many regions will be returned.
        min_positive: if `top` is `None`, only regions with this many cumulative
            infections will be returned.
    """
    s = df.groupby('region').cumulative.max()
    if top:
        return sorted(s.sort_values(ascending=False).index[:top])
    else:
        return sorted(s[s >= min_positive].index)


def _counts(df, region=None, start=None, end=None, min_positive=100):
    """Returns daily cumulative and['raw'].infection counts for a given region.

    Args:
        region: if `None`, countrywide counts will be returned.
        start, end: optional dates used to filter the data, as strings that can
            be converted to pandas timestamps. Only dates with positive
            infection counts will be considered.
        min_positive: If `start` is not specified, the first date on which the
            cumulative count is at least `min_positive` will be used instead.
    """
    if region is None:
        s = df.groupby('date').cumulative.sum()
    else:
        s = df[(df.region == region)].set_index('date').sort_index().cumulative

    s = s[s > 0]
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    else:
        s = s[s >= min_positive]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]

    idx = pd.date_range(s.index.min(), s.index.max()).rename('date')
    s = s.reindex(idx).reset_index()  # allocate rows for missing dates.
    s['new'] = s.cumulative - s.cumulative.shift(fill_value=0)
    return s


def usa_states(top=None, min_positive=100):
    return _regions(_usa(), top, min_positive)


def za_provinces(top=None, min_positive=100):
    return _regions(_za(), top, min_positive)


def usa_counts(state=None, start=None, end=None, min_positive=100):
    return _counts(_usa(), state, start, end, min_positive)


def za_counts(province=None, start=None, end=None, min_positive=100):
    return _counts(_za(), province, start, end, min_positive)


def smooth(df, raw='new', smth='smoothed', std=STD):
    """Adds a column containing smoothed counts to a data frame."""
    df[smth] = df[raw].rolling(
        int(np.ceil(1+6*std)), win_type='gaussian', min_periods=1, center=True
    ).mean(std=std).round()
    return df
