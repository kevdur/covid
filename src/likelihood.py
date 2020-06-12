# ==============================================================================
# Inference functions
# ==============================================================================

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln


def posteriors(df, r, c, col='new', a=1, b=3):
    """Adds inferred parameters of infection-rate posteriors to a data frame.

    Under the assumption that observed new infection counts follow a negative
    binomial distribution, and a conjugate beta success-probability prior, the
    (scaled) daily infection rate is distributed according to a beta prime
    distribution with parameters `alpha` and `beta`. (The success probability's
    posterior is a beta distribution with the same parameters.)

    Args:
        col: the data frame column containing daily new infection counts.
        a, b: parameters of the initial predictive beta prime prior.
    """
    # The posterior parameters are usually increased versions of the predictive
    # prior's, but when a day's count is missing we just use the prior's
    # parameters directly.
    ks = np.nan_to_num(df[col])  # replaces NaNs with 0.
    rs = np.where(np.isnan(df[col]), 0, r)  # 0 if NaN, r otherwise.
    alpha, beta = a + ks[0], b + rs[0]
    alphas, betas = [alpha], [beta]
    for k, r in zip(ks[1:], rs[1:]):
        alpha, beta = alpha/c + k, beta/c + r
        alphas.append(alpha)
        betas.append(beta)
    df['alpha'] = alphas
    df['beta'] = betas
    return df


def lmlhd(dfs, r, c, col='new', a=1, b=3, rlprior=None, clprior=None,
          copy=True):
    """Returns the log marginal likelihood of the model parameters `r` and `c`.

    Args:
        dfs: a data frame or list/tuple of data frames containing counts.
        col: the column containing daily new infection counts.
        a, b: parameters of the initial predictive beta prime prior.
        rlprior, clprior: log density functions to be used as priors on `r` and
            `c` (uniform by default).
    """
    if not isinstance(dfs, list) and not isinstance(dfs, tuple):
        dfs = [dfs]
    if copy:
        dfs = [df.copy() for df in dfs]
    # First we create data frame copies containing posterior columns, and then
    # compute the marginal likelihood over each of these independently.
    dfs = [posteriors(df, r, c, col, a, b) for df in dfs]
    z = sum(_lmlhd(df, r, c, col, rlprior, clprior) for df in dfs)
    if rlprior is not None:
        z += rlprior(r)
    if clprior is not None:
        z += clprior(c)
    return z


def _lmlhd(df, r, c, col='new', rlprior=None, clprior=None):
    df = df[~df[col].isna()]  # ignore days with missing counts.
    # The last line below is only valid when the count is positive. We also take
    # logarithms outside the sum because otherwise the 'where' argument will
    # take the entire return expression into account.
    ls = np.log(df[col], where=(df[col] != 0))
    return (betaln(df.alpha, df.beta)
            - betaln(df.alpha - df[col], df.beta - r)
            - np.where(df[col] != 0, ls + betaln(df[col], r), 0)).sum()


def opt(dfs, col='new', a=1, b=3, rlprior=None, clprior=None):
    """Returns maximum likelihood estimates of the model parameters `r` and `c`.

    The optimised parameters `r` and `c` refer to the failure count of the
    model's negative binomial likelihood function and the variance factor
    introduced by each predictive prior, respectively.

    Args:
        dfs: a data frame or list/tuple of data frames containing counts.
        col: the column containing daily new infection counts.
        a, b: parameters of the initial predictive beta prime prior.
        rlprior, clprior: log density functions to be used as priors on `r` and
            `c` (uniform by default).
    """
    def f(r):
        return _optc(dfs, r, col, a, b, rlprior, clprior, copy=False)[1]

    if not isinstance(dfs, list) and not isinstance(dfs, tuple):
        dfs = [dfs]
    dfs = [df.copy() for df in dfs]  # create copies once, before optimising.
    # We double r until we pass a local minimum, and then optimize the two
    # regions that might contain that minimum separately.
    p, r = 1, 2
    while f(p) > f(r):
        p, r = r, 2*r
    r1, l1 = _cvxsearch(f, p//2, p)
    r2, l2 = _cvxsearch(f, p, r)
    if l1 <= l2:
        return r1, _optc(dfs, r1, col, a, b, rlprior, clprior, copy=False)[0]
    else:
        return r2, _optc(dfs, r2, col, a, b, rlprior, clprior, copy=False)[0]


def _optc(dfs, r, col='new', a=1, b=3, rlprior=None, clprior=None, copy=True):
    def f(c):
        return -lmlhd(dfs, r, c, col, a, b, rlprior, clprior, copy=copy)
    res = minimize_scalar(f, bounds=[1, 100], method='bounded')
    return res.x, res.fun


def _cvxsearch(f, x1, x2):
    """Optimises a convex, integer-domain function using binary search."""
    x = x1 + (x2-x1)//2  # midpoint.
    y1, y, y2 = f(x1), f(x), f(x2)
    if x2-x1 == 1:
        return (x1, y1) if y1 <= y2 else (x2, y2)
    # Recurse on the half-region containing the local minimum.
    if y >= y1:
        return _cvxsearch(f, x1, x)
    elif y >= y2:
        return _cvxsearch(f, x, x2)
    else:
        x1, y1 = _cvxsearch(f, x1, x)
        x2, y2 = _cvxsearch(f, x, x2)
        return (x1, y1) if y1 <= y2 else (x2, y2)
