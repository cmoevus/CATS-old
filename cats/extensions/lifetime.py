# -*- coding: utf8 -*-
"""
Methods related to the lifetime of the content.

Particle:
    dwell_time: number of frames a particle is detected

Particles:
    survival_plot: datapoints of the loss (relative to 1) of particles against time.
    halflife: half-life of the particles
"""
from __future__ import absolute_import, division, print_function
from collections import Counter
import numpy as np


def dwell_time(self):
    """Return the dwell time of the particle."""
    return self['t'].max() - self['t'].min() + 1


def survival_plot(self):
    """Return a survival plot."""
    data = Counter([p.dwell_time() / self.framerate for p in self if p['t'].max() < p.source.length - self.max_blink])
    n, lost = len(self), 0
    x, y, = [0], [1]
    for t, tn in sorted(data.items()):
        lost += tn
        x.append(t)
        y.append((n - lost) / n)
    return x, y


def halflife(self):
    """Return the half-life of the particles based on the survival plot."""
    x, y = self.survival_plot()
    if y[-1] <= 0:
        x, y = x[:-1], y[:-1]
    fit = np.polyfit(x, np.log(y), 1)
    return -np.log(2) / fit[0]


__extension__ = {'Particles': {'survival_plot': survival_plot,
                               'halflife': halflife},
                 'Particle': {'dwell_time': dwell_time}}
