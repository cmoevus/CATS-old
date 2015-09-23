from matplotlib import pyplot as plt
import numpy as np
from math import ceil


def histogram(self, prop='x', binsize=3):
    """
    Draw an histogram of the given property.

    Arguments:
        prop: any spot property (x, y, s, t, i) or 'l', for length of tracks.
        binsize: number of pixels/units per bin.
    """
    if prop == 'l':
        tracks = self.get_tracks('t')
        data = [max(track) - min(track) for track in tracks]
    else:
        tracks = self.get_tracks(prop)
        data = [np.mean(track) for track in tracks]
    bins = int(ceil((max(data) - min(data)) / binsize))
    return plt.hist(data, bins=bins)

__extension__ = {'Experiment': {'histogram': histogram}}
