# -*- coding: utf8 -*-
from collections import Counter
__all__ = ['survival_plot']


def survival_plot(self):
    """Return a survival plot."""
    tracks = self.get_tracks('t')
    data = Counter([(max(track) - min(track)) / self.framerate for track in tracks if max(track) != self.source.length - 1])
    x, y, n, lost = list(), list(), len(tracks), 0
    for t, tn in sorted(data.items()):
        lost += tn
        x.append(t)
        y.append(n - lost)
    return x, y
