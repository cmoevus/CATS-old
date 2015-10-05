# -*- coding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import skimage as ski
import scipy as sp
from ..content.filaments import Filaments, Filament


def constant_filaments(source, blur=0.5, axis=None, keep_unfit=False, max_sigma=5):
    """
    Find filaments between a set of barriers.

    This processor assumes that
        1. The filament spans across the whole length of the given source (given in 'axis')
        2. The filament is perfectly parallel to the axis
        3. The filament never breaks

    Arguments:
        blur: sigma value for smoothing with gaussian blur
        axis: ('x' or 'y') the axis parallel  to the filaments. If None, will consider filaments to be parallel to the shortest dimension of the image.
        keep_unfit: (bool) keep the filaments of which the projection could not be fitted to a Gaussian function (potentially lesser quality detections).
        max_sigma: the maximal standard deviation on a projection of the signal of  filament perpendicular to its length. In other terms, the width of the filament divided by the square root of 2.

    Return:
        Filaments object containing Filament objects with minimal columns plus:
            's': the sigma (lateral 'thickness') of the filament, to be multiplied by sqrt(2) (or more, for example +-3*s for confidence intervals of 95%) for the thickness of the filament
    """
    # A. Make a projection
    image = np.zeros(source.shape)
    for img in source.read():
        image += ski.filters.gaussian_filter(img, blur)
    axis = source.shape.index(min(source.shape)) if axis is None else {'x': 1, 'y': 0}[axis]
    projection = image.sum(axis)

    # B. Find peaks
    peaks, widths = find_peaks(projection)

    # C. Refine by gaussian fitting
    projections = list()
    for p, w in zip(peaks, widths):
        y = projection[w[0]:w[1]]
        x = np.arange(0, len(y))
        a, b, c, d = projection[p], p - w[0], np.std(y), np.min(y)
        try:
            fit, cov = sp.optimize.curve_fit(gaussian, x, y, (a, b, c, d))
            values = (fit[1] + w[0], abs(fit[2]))
            if values[1] <= max_sigma or keep_unfit == True:
                projections.append(values)
        except RuntimeError:
            if keep_unfit == True:
                projections.append((p, c))

    # D. Make Filaments
    filaments = Filaments(processor=constant_filaments, blur=blur, axis=axis, keep_unfit=keep_unfit, max_sigma=max_sigma, sources=source)
    length = image.shape[axis]
    frames = range(source.length)
    dtype = [('x', float), ('y', float), ('l', float), ('a', float), ('s', float), ('t', int)]
    for peak, sigma in projections:
        f = [0, 0, length, 0, sigma]
        f[axis] = peak
        f = np.array([tuple(f + [i]) for i in frames], dtype=dtype).view(Filament)
        f.source = source
        filaments.append(f)
    return filaments


def find_peaks(data):
    """Return a list of local maximas with their width."""
    delta = np.diff(data)
    tangents = list()
    for i in range(0, len(delta) - 1):
        lims = sorted((delta[i], delta[i + 1]))
        if lims[0] < 0 and lims[1] > 0:
            tangents.append(i + 1)
    maxima = list()
    for i in range(1, len(data) - 1):
        if data[i - 1] < data[i] and data[i + 1] < data[i]:
            maxima.append(i)
    peaks = [i for i in tangents if i in maxima]
    widths = list()
    for p in peaks:
        i = tangents.index(p)
        widths.append((tangents[i - 1] if i > 0 else 0, tangents[i + 1]))
    return peaks, widths


def gaussian(x, a, b, c, d):
    """
    Return a Gaussian.

    Arguments:
        x: the values in x
        a: the amplitude
        b: the mean
        c: the standard deviation
        d: the baseline value
    """
    return a * np.e**((-(x - b)**2) / (2 * c**2)) + d

__processor__ = {
    'filaments': {'constant': constant_filaments},
}
