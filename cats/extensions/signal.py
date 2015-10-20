"""
Signal-related functions.

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


@property
def particle_signal(self):
    """Get the signal (area under the curve) of the particle."""
    signals = list()
    for d in self:
        dx, dy = 3 * d['sx'], 3 * d['sy']
        x = np.meshgrid(range(int(d['y'] - dy), int(ceil(d['y'] + dy))), range(int(d['x'] - dx), int(ceil(d['x'] + dx))))
        signals.append(np.sum(gaussian_2d(x, d['a'], d['x'], d['y'], d['sx'], d['sy'])))
    return signals


@particle_signal.setter
def particle_signal(self, value):
    """ Signal is read-only."""
    return AttributeError('signal is read-only')


def particle_snr(self):
    """Find the signal to noise ratio of the particle."""
    return self.signal / ((self['i'] - self['a']) * self['sx'] * self['sy'])


def gaussian_2d(coords, A, x0, y0, sx, sy):
    """
    Draw a 2D gaussian with given properties.

    Arguments:
        A: Amplitude of the Gaussian
        x0: position of the center, in x
        y0: position of the center, in y
        sx: standard deviation in x
        sy: standard deviation in y
    """
    x, y = coords
    return (A * np.exp((x - x0)**2 / (-2 * sx**2) + (y - y0)**2 / (-2 * sy**2))).ravel()


__extension__ = {'Particle': {'signal': particle_signal,
                              'snr': particle_snr}}
