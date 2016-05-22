# -*- coding: utf8 -*-
"""
Copy-paste from extensions/lifetime.py

To make survival plots from list of dwell times instead of objects.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from ..utils import math


def survival_bins(dist, bs_n=1000, binsize=None, ci=0.95, fraction=True):
    """
    Return the bins of a survival plot, with stats.

    Arguments:
        dist: the list of dwell times. Use np.inf if unknown.
        bs_n (int or bool): number of bootstrap iterations. If False or 0, no bootstrap.
        binsize: the number of time units per bin. If None, will only return the times at which events happened.
        ci: confidence interval (float)
        fraction: Return the y axis as a fraction of data left rather than absolute numbers

    Returns:
    x: Time values
    y: Fraction of the data left
    e: error estimate
    """
    # A. Build the distribution of dwell times
    no_bins = False
    if binsize is None:
        diff = sorted(np.diff(sorted(set(dist))))
        binsize = diff[0]  # IS THIS AN ERROR? SHOULD NOT IT BE THAT diff IS SORTED? SO THAT THE BINSIZE IS THE MINIMUM DISTANCE DETECTED?
        no_bins = True
    n_bins = int(max([i for i in dist if i < np.inf]) / binsize) + 1

    # B. Set up the survival distribution(s)
    if bs_n != False:
        bs = math.resample(dist, bs_n)
        shape = (bs_n + 1, n_bins)
    else:
        shape = (1, n_bins)
        bs_n = 0
    survival = np.full(shape, len(dist), dtype=np.int64)
    for i in range(shape[0]):
        d = dist if i == 0 else bs[i - 1]
        for dt in d:
            if dt != np.inf:
                bin_nb = int(dt / binsize)
                survival[i, bin_nb:] -= 1
    if fraction == True:
        survival = survival.astype(np.float64) / len(dist)

    # Get the appropriate (x, y)
    if no_bins == True:
        x = sorted(set([i for i in dist if i < np.inf]))
        y = [survival[0][int(i / binsize)] for i in x]
    else:
        x, y = np.arange(0, round(n_bins * binsize, math.sign_nb(binsize)), binsize), survival[0]  # Believe it or not, the imprecision on n_bins * binsize would cause errors in the number of indexes

    # C. Get the error range and return the whole thing
    if bs_n > 0:
        lb, ub = int((1 - ci) * shape[0]), int(ci * shape[0])
        error = list()
        for i in x:
            d = sorted(survival[:, int(i / binsize)])
            error.append((d[lb], np.mean(d), d[ub]))
        return x, y, error
    else:
        return x, y
