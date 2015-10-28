# -*- coding: utf8 -*-
"""Basic mathematical functions used for fitting and such."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp


def gaussian(x, A, m, s):
    """
    Gaussian probability density function.

    x: the values in x
    A: the amplitude
    m: the mean
    s: the standard deviation
    """
    return A * np.e**((-(x - m)**2) / (2 * s**2))


def gamma_cdf(x, k, T):
    """
    Cumulative Gamma distribution function.

    x: the values in x
    k: the shape factor
    T: the scale factor
    """
    return sp.stats.gamma.cdf(x, k, scale=T)


def gaussian_2d(coords, A, x0, y0, sx, sy):
    """
    2D Gaussian probability density function.

    A: Amplitude of the Gaussian
    x0: position of the center, in x
    y0: position of the center, in y
    sx: standard deviation in x
    sy: standard deviation in y
    """
    x, y = coords
    return (A * np.exp((x - x0)**2 / (-2 * sx**2) + (y - y0)**2 / (-2 * sy**2))).ravel()
