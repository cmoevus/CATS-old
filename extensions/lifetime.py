# -*- coding: utf8 -*-
"""
Methods related to the lifetime of the content.

Particle:
    dwell_time: number of frames a particle is detected

Particles:
    survival: datapoints of the loss (relative to 1) of particles against time.
    halflife: half-life of the particles
"""
from __future__ import absolute_import, division, print_function
from collections import Counter
import numpy as np
from ..utils import math


def dwell_time(self):
    """Return the dwell time of the particle."""
    return self['t'].max() - self['t'].min() + 1


def survival_bins(self, bs_n=1000, binsize=None, ci=0.95, fraction=True):
    """
    Return the bins of a survival plot, with stats.

    Arguments:
        bs_n (int or bool): number of bootstrap iterations. If False or 0, no bootstrap.
        binsize: the number of time units per bin. If None, will only return the times at which events happened.
        ci: confidence interval (float)
        fraction: Return the y axis as a fraction of data left rather than absolute numbers

    Returns:
    x: Time values
    y: Fraction of the data left
    e: error estimate
    
    """
    # Find the cutoff of the data, i.e. the shortest dataset.
    cutoff = np.inf
    for source in self.sources:
        v = source.length - self.max_blink
        if v < cutoff:
            cutoff = v

    # A. Build the distribution of dwell times
    dist, last = list(), 0
    for p in self:
        dt = p.dwell_time() if p['t'].max() < cutoff else np.inf
        dist.append(dt)
        if dt > last and dt != np.inf:
            last = dt
    no_bins = False
    if binsize is None:
        diff = np.diff(sorted(set(dist)))
        # I DO NOT UNDERSTAND THIS ANYMORE. CHANGED FOR THE BEST?
        # binsize = diff[0] if diff[-1] != 0 else diff[1]
        if len(diff) > 0:
            binsize = diff[0]
        else:
            binsize = 1
        no_bins = True
    n_bins = int(last / binsize) + 1

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
        x = sorted(set([i for i in dist if i <= cutoff]))
        y = [survival[0][int(i / binsize)] for i in x]
    else:
        x, y = np.arange(0, round(n_bins * binsize, math.sign_nb(binsize)), binsize), survival[0]  # Believe it or not, the imprecision on n_bins * binsize would cause errors in the number of indexes

    # THIS NEEDS TO LEAVE
    x = [i / self.framerate for i in x]
    binsize /= self.framerate
    #

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


def survival(self, fraction=True):
    """Survival plot that only returns the bins at which events occured."""
    data = Counter(p.dwell_time() for p in self if p['t'].max() < p.source.length - self.max_blink)
    n, lost = len(self), 0
    x, y, = [0], [n]
    for t, tn in sorted(data.items()):
        lost += tn
        x.append(t)
        y.append(n - lost)
    if fraction == True:
        y = [i / n for i in y]

    # THIS NEEDS TO LEAVE
    x = [i / self.framerate for i in x]
    #

    return x, y


def halflife(self, ci=0.95):
    """Return the half-life of the particles based on the survival plot."""
    x, y, e = self.survival_bins(ci=ci)
    y = [i[1] for i in e]  # Use the mean value rather than sample value.
    if y[-1] <= 0:
        x, y = x[:-1], y[:-1]
    fit = np.polyfit(x, np.log(y), 1)
    return -np.log(2) / fit[0]


__extension__ = {'Particles': {'survival': survival,
                               'survival_bins': survival_bins,
                               'halflife': halflife},
                 'Particle': {'dwell_time': dwell_time}}
