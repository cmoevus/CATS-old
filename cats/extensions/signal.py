"""
Signal-related extensions.

The module adds the following methods:
    - snr(): signal-to-noise ratio of the particle
    - signal(): total signal (area under the curve), omitting the noise.
to the following objects:
    - Particle
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import ceil
import scipy as sp
from ..utils.math_functions import gamma_cdf
# Method 1:
# A. Get the noise in the image
# B. Find the diameter of the signal (3 * sigma for 95% of the data)
# C. Cut a ROI of that signal
# D. Get the value of that ROI - the noise of the image.
#
# Method 2:
# A. Find the diameter of the signal (3 * sigma for 95% of the data)
# B. Cut a ROI of that signal
# C. Fit a Gaussian in x and y (just like processor), get B, x, y, sx, sy, A.
# D. Return (sum of gaussian2d(shape, *fit)) / (fit[noise] * shape)


def particle_signal(self):
    """Get the signal (area under the curve) of the particle."""
    from cats.sources import ROI
    signals = list()
    for d in self:
        dx, dy = 3 * d['sx'], 3 * d['sy']
        data = ROI(self.source, x=(d['x'] - dx, d['x'] + dx), y=(d['y'] - dy, d['y'] + dy)).get(d['t'])
        signals.append(data.sum())
        # x = np.meshgrid(range(int(d['y'] - dy), int(ceil(d['y'] + dy))), range(int(d['x'] - dx), int(ceil(d['x'] + dx))))
        # signals.append(np.sum(gaussian_2d(x, d['i'], d['x'], d['y'], d['sx'], d['sy'])))
    return signals


def particle_snr(self):
    """Find the signal to noise ratio of the particle."""
    return self.signal / ((self['i'] - self['a']) * self['sx'] * self['sy'])


def blinking_cdf(self, params=True):
    """
    Return the cumulative distribution function of the blinking behavior of the particles.

    The blinking is estimated as the time between two detections.
    Somehow, it can be fitted with a Gamma distribution, but not a Poisson or such. At least so far...

    Argument:
        params (bool): if True, returns the fitted parameters of the Gamma distribution (shape and scale parameters) as (fit, cov), as scipy.optimize.curve_fit(). If False, return the histogram as (values, bins + rightmost), as numpy.histogram()
    """
    ts = [i - 1 for p in self for i in np.diff(sorted(p['t']))]
    bins = range(0, self.max_blink)
    h = np.histogram(ts, normed=True, bins=bins)
    cdf = [np.sum(h[0][0:j + 1]) for j in range(len(h[0]))]
    if params == True:
        return sp.optimize.curve_fit(gamma_cdf, h[1][:-1], cdf)
    else:
        return cdf, h[1][:-1]


__extension__ = {'Particles': {'blinking_cdf': blinking_cdf},
                 'Particle': {'signal': particle_signal,
                              'snr': particle_snr}}
