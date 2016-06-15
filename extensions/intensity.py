# -*- coding: utf8 -*-
"""
Analyse the intensity of content against time.

Supports smoothing by moving window, selection of a time frame and noise substraction.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from skimage import draw
from ..utils.closest_value import get_closest_value
from ..utils.slicify import slicify
from math import ceil


def intensity(self, size=None, t=None, smoothing_window=None, substract_noise=True):
    """
    Mean intensity at the position of the particle along time.

    Arguments:
        size <int, float, 2-tuple>: radius of the ellipse to consider for intensity, in pixels. Centered at the center of the particle. If None, uses the mean sx and sy values of the particle.
        t <2,3-tuple, slice, int>: the timepoints at which to collect the intensity information. As compatible with cats.utils.slicify. If 'None', uses the whole length of the dataset.
        smoothing_window <int>: The number of datapoints to average to smooth the output
        substract_noise <bool>: Whether to substract the surrounding value of noise from the intensity.
    """
    noise_distance = 2  # Extra diameter to get noise close to content
    T, I = list(), list()

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

        # Get the xy position and data at exact or closest time
        roi = draw.ellipse(int(self.y[index]), int(self.x[index]), sy, sx, shape=self.source.shape)
        spot = self.source.get(t)[roi]
        intensity = spot.mean()

        # Remove noise
        if substract_noise == True:
            noise = draw.ellipse_perimeter(int(self.y[index]), int(self.x[index]), sy + noise_distance, sx + noise_distance, shape=self.source.shape)
            noise = self.source.get(t)[noise]
            intensity -= noise.mean()

        T.append(t)
        I.append(intensity)

    if smoothing_window is not None and smoothing_window > 1:
        for t in range(0, len(I)):
            j, k = max(t - smoothing_window, 0), min(t + smoothing_window, len(I))
            I[t] = np.median(I[j:k])

    return T, I


__extension__ = {
    "Particle": {"intensity": intensity}
}
