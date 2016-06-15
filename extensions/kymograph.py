# -*- coding: utf8 -*-
"""Find the number of photobleaching steps for a given particle."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from skimage import draw
from ..utils.closest_value import get_closest_value
from ..utils.slicify import slicify
from math import ceil


def kymograph(self, size=None, t=None):
    """Return a kymograph of the particle."""
    #
    # This moves with the fluorophore, so it is NOT a kymograph...
    #
    P = list()

    # Treat time
    ts = slicify(t, slice(0, self.source.length, 1))

    # Treat size
    if size is None:
        sx, sy = int(ceil(np.mean(self.sx))), int(ceil(np.mean(self.sy)))
    elif type(size) in (int, float):
            sx, sy = int(ceil(size)), int(ceil(size))
    elif '__iter__' in dir(size):
        sx, sy = int(ceil(size[0])), int(ceil(size[1]))

    # Get data
    for t in range(ts.start, ts.stop, ts.step):
        # Get the exact or closest position for given time
        if t not in self.t:
            index = np.where(self.t == get_closest_value(t, self.t))[0]
        else:
            index = np.where(self.t == t)[0]
        rx, ry = slice(max(0, int(self.x[index] - sx)), min(self.source.shape[1], int(self.x[index] + sx))), slice(max(0, int(self.y[index] - sy)), min(self.source.shape[0], int(self.y[index] + sy)))
        P.append(np.mean(self.source.get(t)[ry, rx], axis=1))

    return P


__extension__ = {
    "Particle": {"kymograph": kymograph}
}
