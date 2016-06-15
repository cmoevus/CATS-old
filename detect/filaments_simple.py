# -*- coding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import skimage as ski
import scipy as sp
from ..content.filaments import Filaments, Filament


def constant_filaments(source, blur=0.5, axis=None, keep_unfit=False):
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

    Return:
        Filaments object containing Filament objects with minimal columns plus:
            's': the sigma (lateral 'thickness') of the filament, to be multiplied by sqrt(2) (or more, for example +-3*s for confidence intervals of 95%) for the thickness of the filament
    """
    # A. Make a projection
    image = np.zeros(source.shape)
    for img in source.read():
        image += ski.filters.gaussian_filter(img, blur)
    axis = source.shape.index(min(source.shape)) if axis is None else {'x': 1, 'y': 0}[axis]
    image -= image.mean()
    image *= np.where(image < 0, 0, 1)
    projection = image.sum(axis)

    # B. Find peaks
    peaks, widths = find_peaks(projection)

    # C. Refine by gaussian fitting
    projections = list()
    for p, w in zip(peaks, widths):
        y = projection[w[0]:w[1]]
        datapoints = w[1] - w[0]
        x = np.arange(0, datapoints)
        A, m, s = projection[p], p - w[0], np.std(y)
        if datapoints == 2:
            func = normal_dist
            p0 = (m, s)
        elif datapoints >= 3:
            func = gaussian
            p0 = (A, m, s)
        else:
            continue
        try:
            fit, cov = sp.optimize.curve_fit(func, x, y, p0)
            if datapoints == 2:
                values = (A, fit[0] + w[0], abs(fit[1]))
            else:
                values = (fit[0], fit[1] + w[0], abs(fit[2]))
            if 0 <= values[1] <= projection.shape[not axis]:
                projections.append(values)
        except RuntimeError:
            if keep_unfit == True:
                projections.append((A, p, s))

    # D. Remove noise
    # A = [p[0] for p in projections]
    # mA, sA = np.mean(A), np.std(A)
    # print(mA, sA, mA - 3 * sA, mA + 3 * sA)
    # print(A)
    #
    # return projections, projection, A

    # D. Make Filaments
    filaments = Filaments(processor=constant_filaments, blur=blur, axis=axis, keep_unfit=keep_unfit, sources=source)
    length = image.shape[axis]
    frames = range(source.length)
    dtype = [('x', float), ('y', float), ('l', float), ('a', float), ('s', float), ('t', int)]
    for A, m, s in projections:
        f = [0, 0, length, 0, s]
        f[axis] = m
        f = np.array([tuple(f + [i]) for i in frames], dtype=dtype).view(Filament)
        f.source = source
        filaments.append(f)
    return filaments


def find_peaks(data):
    """Return a list of local maximas with their width."""
    # A. Find peaks
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

    # B. Find widths
    minima = list()
    for i in range(1, len(data) - 1):
        if data[i - 1] >= data[i] and data[i + 1] >= data[i]:
            minima.append(i)
    minima = np.array(minima)
    widths = list()
    for p in peaks:
        limits = np.where(minima > p, True, False)
        tr = list(limits).index(1)
        widths.append((minima[tr - 1] if tr > 0 else 0, minima[tr]))

    return peaks, widths


def gaussian(x, A, m, s):
    """
    Return a Gaussian.

    Arguments:
        x: the values in x
        A: the amplitude
        m: the mean
        s: the standard deviation
        n: the baseline noise
    """
    return A * np.e**((-(x - m)**2) / (2 * s**2))


def normal_dist(x, m, s):
    return np.e**((-(x - m)**2) / (2 * s**2)) / (s * np.sqrt(np.pi * 2))


__processor__ = {
    'filaments': {'constant': constant_filaments},
}
