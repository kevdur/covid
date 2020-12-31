# ==============================================================================
# Plotting functions
# ==============================================================================

import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd
from scipy.stats import betaprime
from src.data import GAMMA, STD, smooth
from src.likelihood import posteriors

ZA_EVENTS = [
    (pd.Timestamp(2020, 3, 23), 'Lockdown announced'),
    (pd.Timestamp(2020, 3, 27), 'Level 5'),
    (pd.Timestamp(2020, 5, 1), 'Level 4'),
    (pd.Timestamp(2020, 6, 1), 'Level 3'),
    (pd.Timestamp(2020, 8, 18), 'Level 2'),
    (pd.Timestamp(2020, 9, 21), 'Level 1'),
    (pd.Timestamp(2020, 12, 29), 'Level 3b')
]


def event_lines(events):
    """Plots vertical lines at a set of given dates."""
    return hv.Overlay([hv.VLine(d, label=l).opts(color='grey', line_width=1)
                       for d, l in events])


def smoothed(df, std=STD, title='New infections'):
    """Plots raw and smoothed new infection counts."""
    df = df.copy()
    smooth(df, raw='new', smth='smoothed', std=std)
    df['raw'] = df.new
    return df.hvplot(
        x='date', y=['raw', 'smoothed'], legend='top_left'
    ).opts(title=title) \
        * df.hvplot.scatter(x='date', y=['raw', 'smoothed'], s=20)


def smoothed_diff(df, raw='new', gamma=GAMMA, std=STD,
                  title='Infections-period bound'):
    """Plots the difference between smoothed counts and an implicit gamma bound.

    The value of the infectious-period parameter `gamma` implies a lower bound
    on the rate at which the number of new infections can decrease. This plot
    shows the difference between the smoothed counts and this implicit bound.
    The difference will ideally remain non-negative.
    """
    df = df.copy()
    smooth(df, raw=raw, smth='smoothed', std=std)
    # We floor the bound to make it more realistic (and lenient), since
    # otherwise, for example, a decoy from 1 to 0 would not be allowed.
    df['γ bound'] = np.floor(df.smoothed.shift()*np.exp(-gamma))
    df['difference'] = df.smoothed - df['γ bound']
    return df.hvplot.line(
        x='date', y=['smoothed', 'γ bound', 'difference'],
        line_dash=['dashed', 'dashed', 'solid'], legend='top_left'
    ).opts(title=title) \
        * hv.HLine(np.exp(-gamma))


def smoothed_ratio(df, raw='new', gamma=GAMMA, stds=[STD],
                   title='Inter-day ratio'):
    """Plots a smoothed-count ratio and its implicit gamma bound.

    The value of the infectious-period parameter `gamma` implies a lower bound
    on the rate at which the number of new infections can decrease. This plot
    shows this rate, which will ideally remain above the bound.
    """
    df = df.copy()
    bound = np.exp(-gamma)
    mn, mx = bound, bound
    smths = [f'smoothed {std}' for std in stds]
    ratios = [f'σ = {std:.1f}' for std in stds]
    for std, smth, ratio in zip(stds, smths, ratios):
        smooth(df, raw=raw, smth=smth, std=std)
        # We add one to the observed value k to account for the fact that
        # according to the bound, the least tolerable value might be between k
        # and k+1 (i.e., for the same reason we applied the floor operation in
        # the difference-plot function).
        df[ratio] = (df[smth] + 1)/df[smth].shift()
        mn, mx = min(mn, df[ratio].min()), max(mx, df[ratio].max())
    ylim = (mn - 0.1*(mx-mn), mx + 0.1*(mx-mn))
    return df.hvplot.line(
        x='date', y=ratios, ylim=ylim, legend='top_right'
    ).opts(title=title) \
        * hv.HLine(bound)


def _posterior(df, r, c, col='new', a=1, b=3, gamma=GAMMA):
    df = df.copy()
    posteriors(df, r, c, col, a, b)
    df['data'] = df[col]
    # Posterior on lambda, the daily rate of infection.
    df['posterior'] = r*betaprime.median(df.alpha, df.beta)
    df['pos5'] = r*betaprime.ppf(0.05, df.alpha, df.beta)
    df['pos95'] = r*betaprime.ppf(0.95, df.alpha, df.beta)
    df['r'] = (np.log(df.posterior.shift(-1)) - np.log(df.posterior))/GAMMA + 1
    df['r5'] = (np.log(df.pos5.shift(-1)) - np.log(df.posterior))/GAMMA + 1
    df['r95'] = (np.log(df.pos95.shift(-1)) - np.log(df.posterior))/GAMMA + 1
    return df[~df.r.isna()]


def posterior(df, r, c, col='new', a=1, b=3, gamma=GAMMA,
              title='Inferred daily infection rate'):
    """Plots estimated daily infection rates."""
    df = _posterior(df, r, c, col, a, b, gamma)
    return df.hvplot.line(
        x='date', y=['data', 'posterior'], legend='top_left'
    ).opts(title=title) \
        * df.hvplot.area(x='date', y='pos5', y2='pos95', color='orange',
                         alpha=0.2)


def posterior_diff(df, r, c, col='new', a=1, b=3, gamma=GAMMA,
                   title='Infectious-period bound'):
    """Plots the difference between the inferred rate and a gamma bound.

    This plot provides an indication of whether or not the inferred rates are
    sufficiently smooth; see `smoothed_diff` -- which does the same for
    smoothed, observed counts -- for more details.
    """
    df = _posterior(df, r, c, col, a, b, gamma)
    df['γ bound'] = df.posterior.shift()*np.exp(-gamma)
    df['difference'] = df.posterior - df['γ bound']
    return df.hvplot.line(
        x='date', y=['posterior', 'γ bound', 'difference'],
        line_dash=['dashed', 'dashed', 'solid'], legend='top_left'
    ).opts(title=title) \
        * hv.HLine(np.exp(-gamma))


def posterior_ratio(df, r, c, col='new', a=1, b=3, gamma=GAMMA,
                    title='Inter-day ratio'):
    """Plots an inferred-rate ratio and its implicit gamma bound.

    This plot provides an indication of whether or not the inferred rates are
    sufficiently smooth by considering their rate of change; see
    `smoothed_ratio` -- which does the same for smoothed, observed counts -- for
    more details.
    """
    df = _posterior(df, r, c, col, a, b, gamma)
    bound = np.exp(-gamma)
    df['ratio'] = df.posterior/df.posterior.shift()
    # mn, mx = min(bound, df.ratio.min()), max(bound, df.ratio.max())
    # ylim = (mn - 0.1*(mx-mn), mx + 0.1*(mx-mn))
    return df.hvplot.line(
        x='date', y='ratio', legend='top_right'
    ).opts(title=title) \
        * hv.HLine(bound)


def reproduction(df, r, c, col='new', a=1, b=3, gamma=GAMMA,
                 title='Effective reproduction number'):
    """Plots estimated reproduction numbers."""
    df = _posterior(df, r, c, col, a, b, gamma)
    return df[df.r >= 0].hvplot.line(
        x='date', y='r', legend='top_right', ylim=(0, df.r.max()+0.5)
    ).opts(title=title) \
        * df[df.r >= 0].hvplot.scatter(
            x='date', y='r', c='r', cmap='bkr', colorbar=False
    ).redim.range(r=(0, 1.5)) \
        * df.hvplot.area(x='date', y='r5', y2='r95', alpha=0.2) \
        * hv.HLine(1)
