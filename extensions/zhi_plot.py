from matplotlib import pyplot as plt
import numpy as np
from math import floor, ceil


def zhi_plot(self):
    """Plot lifetime against time of initial binding."""
    binsize = int(ceil(self.source.length / (10 * self.framerate)))
    exp_len = int(ceil(self.source.length / self.framerate))

    bins = [[] for t in range(0, exp_len, binsize)]
    tracks = self.get_tracks('t')
    for track in tracks:
        frames = [t for t in track]
        m = min(frames)
        dt = (max(frames) - m) / self.framerate
        bins[int(floor((m / self.framerate) / binsize))].append(dt)

    # All the dwell times
    xs = list()
    ys = list()
    for i, b in enumerate(bins):
        xs.extend([i * binsize for d in b])
        ys.extend(b)
    plt.plot(xs, ys, 'g.', label='Calculated dwell times')

    # Mean dwell tims
    y = [np.mean(b) for b in bins]
    x = range(0, exp_len, binsize)
    plt.plot(x, y, 'ro', label='Mean dwell time: {0:.2f}s'.format(np.mean(ys)))
    plt.ylabel('Mean dwell time (s)')
    plt.yscale('log')
    plt.xlabel('Binned time of initial binding (s)')
    plt.legend()
    plt.figtext(0.82, 0.12, 'N={0}'.format(len(tracks)))
    plt.savefig('zhi_plot.png')

__extension__ = {'Experiment': {'zhi_plot': zhi_plot}}
