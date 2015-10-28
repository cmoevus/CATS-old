# -*- coding: utf8 -*-
"""Filters based on signal information."""
from __future__ import absolute_import, division, print_function
import numpy as np
from ..extensions import signal
import scipy as sp


def blinking_thresholding(self, sl=0.01):
    """
    Split tracks where the time between two detection (blinking) is greater than that allowed by the distribution of blinkings.

    sl: (float: 0 to 1) the significance interval used to threshold the data. The track will be split at any time between two detection if that time is superior to the time at which sl is reached in the cumulative distribution of blinking.
    """
    # Get the blinking cdf
    fit, cov = self.blinking_cdf(params=True)
    dist = sp.stats.gamma(fit[0], scale=fit[1])
    threshold = 1 - sl
    filtered = type(self)(**self)
    for p in self:
        cut = 0
        p.sort(order='t')
        print(p['t'])
        for t, d in enumerate(np.diff(p['t']) - 1):
            if dist.cdf(d) > threshold:
                print(d, dist.cdf(d))
                filtered.append(p[cut:t + 1])
                cut = t + 1
        filtered.append(p[cut:])

    if 'filters' not in dir(filtered):
        filtered.filters = []
    filtered.filters += [blinking_thresholding, {'sl': sl}]
    return filtered

__filter__ = {'Particles': blinking_thresholding}
