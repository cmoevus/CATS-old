# -*- coding: utf8 -*-
"""Implement useful plotting functions."""
import numpy as np
from matplotlib import pyplot as plt


def errorhist(hist, bins, errors, **kwargs):
    """
    Draw a histogram  with error bars from the output the histogram function of CATS.

    Arguments:
        hist, bins, errors: output of content.histogram()
        Extra kwargs will be passed directly to plt.bar
    """
    widths = np.diff(bins)
    low_err = [hist[i] - errors[i][0] for i in range(len(hist))]
    up_err = [errors[i][2] - hist[i] for i in range(len(hist))]
    return plt.bar(bins[:-1], height=hist, width=widths, yerr=[low_err, up_err], **kwargs)


def herrorhist(hist, bins, errors, **kwargs):
    """
    Draw a vertical histogram  with error bars from the output of np.histogram or the histogram function of CATS.

    Arguments:
        hist
        bins
        errors
        Extra kwargs will be passed directly to plt.bar
    """
    heights = np.diff(bins)
    low_err = [hist[i] - errors[i][0] for i in range(len(hist))]
    up_err = [errors[i][2] - hist[i] for i in range(len(hist))]
    return plt.barh(bins[:-1], width=hist, height=heights, xerr=[low_err, up_err], **kwargs)
