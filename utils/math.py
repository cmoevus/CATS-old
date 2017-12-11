# -*- coding: utf8 -*-
"""Basic mathematical functions used for fitting and such."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp
import math
import random
import pandas as pd


def normal(x, mu, sigma):
    """Calculate values from the normal distribution."""
    return np.exp((x - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi * sigma**2))


def noisy_gaussian(x, A, m, s, b):
    """Gaussian probability density function in a non-zero background.

    Parameters:
    -----------
        x
            the values in x
        A
            the amplitude
        m
            the mean
        s
            the standard deviation
        b
            the underlying noise

    """
    return A * np.e**((-(x - m)**2) / (2 * s**2)) + b


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


def exp_decay(x, N, k):
    """Exponential decay."""
    return N * np.e**(-k * x)


def two_exp_decay(x, a, k1, k2):
    """Sum of two exponential decays."""
    return a * np.exp(-k1 * (x)) + (1 - a) * np.exp(-k2 * (x))


def resample(dist, n=1000):
    """
    Resample the given distribution into n distributions of the same size.

    dist: the original distribution (list)
    n: number of distributions

    returns an ndarray of resample distributions

    """
    return np.array([random.choice(dist) for i in range(len(dist) * n)]).reshape(n, len(dist))


def resample_df(df, n=5000):
    """
    Resample the given pandas DataFrame into n DataFrames of the same size.

    Parameters:
    -----------
    df: pandas.DataFrame
        the original DataFrames
    n: int
        number of resamplings to generate to generate

    Returns:
    --------
    resampled: pandas.DataFrames
        A DataFrame with each resampled input DataFrame numbered in the column 'resample_id'. The first dataset (resample_id = 0) is the original one.

    """
    df = df.copy()
    df['resample_id'] = 0
    dfs = [df]
    for i in range(1, n):
        rdf = df.sample(frac=1, replace=True)
        rdf['resample_id'] = i
        dfs.append(rdf)
    return pd.concat(dfs)


def sign_nb(x):
    """
    Return the number of significant digits (negative if int).

    To be used with round:
        round(y, sign_nb(x))
    """
    return -int(math.floor(np.log10(x)))


def line_from_points(pt1, pt2):
    """Return the m and b parameters of the line mx + b going through (x1, y1) and (x2, y2)."""
    x1, y1 = pt1
    x2, y2 = pt2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b
