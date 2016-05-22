"""Build an histogram of positions."""
from __future__ import absolute_import, division, print_function
from math import ceil
from ..utils.math import resample
import numpy as np


def histogram(self, prop='x', binsize=None, bins=10, mean=True, bs_n=1000, ci=0.95, **kwargs):
    """
    Draw an histogram of the given property.

    Arguments:
        prop: any detection property (x, y, s, t, i, etc.) or 'l', for length of tracks.
        binsize: number of units per bin.
        bins: If binsize is not set, defaults to this argument. Behaves like numpy.histogram
        mean: if True, the mean value of each 'content' will be plotted. If False, the value of each detection of each 'content' will be plotted.
        bs_n (int or bool): number of bootstrap iterations. If False or 0, no bootstrap.
        ci: confidence interval (float)
        Additional arguments will be passed to numpy.histogram.

    Returns:
        hist
        bins
        if bs_n > 0: stats
    """
    # A. Build the original distribution
    if prop == 'l':
        data = [p.dwell_time() for p in self]
    elif mean == True:
        data = [p[prop].mean() for p in self]
    else:
        data = [i for p in self for i in p[prop]]

    # B. Figure out the bins
    if binsize is not None:
        if 'range' in kwargs:
            bins = int(ceil((kwargs['range'][1] - kwargs['range'][0]) / binsize))
        else:
            bins = int(ceil((max(data) - min(data)) / binsize))

    # C. Get the initial histogram
    kwargs = dict([(k, v) for k, v in kwargs.items() if k not in ['prop', 'binsize', 'mean', 'bs_n', 'ci']])
    hist, bins = np.histogram(data, bins=bins, **kwargs)

    # D. Bootstrap and get the stats
    if bs_n != False:
        # D1. Get all the resampled histograms
        hists = list()
        for d in resample(data, bs_n):
            hists.append(np.histogram(d, bins=bins, **kwargs)[0])
        hists.append(hist)
        hists = np.array(hists)

        # D2. Get the stats
        lb, ub = int((1 - ci) * hists.shape[0]), int(ci * hists.shape[0])
        stats = list()
        for i in range(0, hists.shape[1]):
            d = sorted(hists[:, i])
            stats.append((d[lb], np.mean(d), d[ub]))
        return hist, bins, stats
    return hist, bins

__extension__ = {'Particles': {'histogram': histogram}}
