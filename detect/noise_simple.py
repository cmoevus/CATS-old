# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp
from math import ceil
from ..content.noise import Noise
from ..utils.math import gaussian


def find_noise(source, window=None):
    """
    Find the noise in the given set of images.

    Arguments:
        source: the source files (sources.Images/ROI object)
        window: the size of the window, in frames, to average noise in. If None, the noise value will be calculated on one window containing all images.

    Returns a Noise object with the minimal fields.
    """
    if window is None:
        window = source.length - 1
    lower_lim, noises = 0, list()
    for upper_lim in range(window, source.length + 1, window):
        # A. Build the mean image
        img = np.zeros(source.shape)
        for f in range(lower_lim, upper_lim):
            img += source.get(f) / window

        # B. Fit
        try:
            fit, cov = fit_noise(img)
        except ValueError:
            pass

        # C. Write down
        for f in range(lower_lim, upper_lim):
            noises.append((f, fit[1], fit[2]))

        # D. Update for the next loop
        lower_lim = upper_lim

    dtype = [('t', int), ('m', float), ('s', float)]
    N = np.array(noises, dtype=dtype).view(Noise)
    N.processor = find_noise
    N.window = window
    N.source = source
    return N


def fit_noise(img):
    """Fit the noise on the image."""
    # A. Use Scott's normal reference rule for the bin width and plot the histogram
    data = img.ravel()
    binwidth = 3.5 * np.std(data) / len(data)**(1 / 3)
    bins = ceil((data.max() - data.min()) / binwidth)
    y, x = np.histogram(data, bins)
    dx = (x[1] - x[0]) / 2
    x_centered = [i + dx for i in x][:-1]

    # B. Fit the histogram to the expected normal distribution
    p0 = y.max(), x_centered[np.where(y == y.max())[0][0]], np.std(data)
    fit, cov = sp.optimize.curve_fit(gaussian, x_centered, y, p0=p0)

    return fit, cov


__processor__ = {'noise': {'fit': find_noise}}
