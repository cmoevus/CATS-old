# -*- coding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import Counter


def survival_plot(self):
    """Return a survival plot."""
    data = Counter([(max(p['t']) - min(p['t'])) / float(self.framerate) for p in self if p['t'].max() < p.source.length - self.max_blink])
    n, lost = len(self), 0
    x, y, = [0], [1]
    for t, tn in sorted(data.items()):
        lost += tn
        x.append(t)
        y.append((n - lost) / n)
    return x, y

__extension__ = {'Particles': {'survival_plot': survival_plot}}
