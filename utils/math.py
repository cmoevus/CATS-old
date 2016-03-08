# -*- coding: utf8 -*-
"""Basic mathematical functions used for fitting and such."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp
import math
import random


def gaussian(x, A, m, s):
    """
    Gaussian probability density function.

    x: the values in x
    A: the amplitude
    m: the mean
    s: the standard deviation
    """
    return A * np.e**((-(x - m)**2) / (2 * s**2))


def gamma_cdf(x, k, T, l):
    """
    Cumulative Gamma distribution function.

    x: the values in x
    k: the shape factor
    T: the scale factor
    l: the location factor
    """
    return sp.stats.gamma.cdf(x, k, scale=T, loc=l)


def gamma_pdf(x, k, T, l):
    """
    Cumulative Gamma distribution function.

    x: the values in x
    k: the shape factor
    T: the scale factor
    l: the location factor
    """
    return sp.stats.gamma.pdf(x, k, scale=T, loc=l)


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


def exp_decay(x, k, b):
    return np.e**(-k * x + b)


def resample(dist, n=1000):
    """
    Resample the given distribution into n distributions of the same size.

    dist: the original distribution (list)
    n: number of distributions

    returns an ndarray of resample distributions
    """
    return np.array([random.choice(dist) for i in range(len(dist) * n)]).reshape(n, len(dist))


def sign_nb(x):
    """
    Return the number of significant digits (negative if int).

    To be used with round:
        round(y, sign_nb(x))
    """
    return -int(math.floor(np.log10(x)))
