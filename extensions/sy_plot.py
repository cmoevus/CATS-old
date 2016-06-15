from matplotlib import pyplot as plt
from math import ceil
import numpy as np


def sy_plot(self, binsize=3):
    """Draw a SY (Stacked Ys) plot based on the tracks."""
    # Define the limits in X
    lims = (0, max([s for t in self.get_tracks('x') for s in t]))

    # Create a dic of position -> lengths of tracks
    bins = np.arange(0, int(ceil(lims[1])), binsize)
    y = dict()
    for i in range(0, len(bins)):
        y[i] = list()

    n = 0
    for track in self.get_tracks(['x', 't']):
        frames = [spot['t'] for spot in track]
        length = max(frames) - min(frames) + 1
        position = np.mean([spot['x'] for spot in track])
        b = max([i for i, b in enumerate(bins) if position - b >= 0])
        y[b].append(np.log(length))
        n += 1

    # Build a masked array to feed plt.pcolor with
    max_length = max([len(t) for t in y.values()])
    C = np.ma.zeros((len(bins), max_length))
    mask = np.ones((len(bins), max_length), dtype=np.bool)
    for x, data in y.items():
        datapoints = len(y[x])
        # Unmask
        mask[x, 0:datapoints] = False
        # Fill the column
        C[x, 0:datapoints] = sorted(y[x])

    C.mask = mask
    plt.figure()
    plt.pcolor(C.T)
    cb = plt.colorbar()
    cb.set_label('Length of the interaction (frames)')
    lengths = [v for l in y.values() for v in l]
    cb_ticks = np.linspace(min(lengths), max(lengths), 10)
    cb.set_ticks(cb_ticks)
    cb.set_ticklabels([int(round(np.e**v, 0)) for v in cb_ticks])
    ticks = range(0, len(bins), 2)
    plt.xticks(ticks, [int(bins[i]) for i in ticks])
    plt.figtext(0.25, 0.87, ha='right', va='top', s='N={0}'.format(n))
    plt.xlabel('Position on DNA (pixel)')
    plt.ylabel('Number of interactions')
    plt.savefig('sy_plot.png')
    plt.close()

    return bins, y

__extension__ = {'Experiment': {'sy_plot': sy_plot}}
