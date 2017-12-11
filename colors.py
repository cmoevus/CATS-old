# -*- coding: utf8 -*-
"""Simple functions to pick colors."""
from __future__ import absolute_import, division, print_function
import numpy as np
from random import randrange
from colorsys import hls_to_rgb, rgb_to_hls
from math import ceil

import cats.images


def random():
    """Return a random yet curated RGB color as a 3-tuple of int[0, 255]."""
    return tuple(int(round(i * 255, 0)) for i in hls_to_rgb(randrange(0, 360) / 360, randrange(50, 90, 1) / 100, randrange(30, 80, 10) / 100))


def get_colors(n=72,
            #    base_hues=(120, 300, 240, 0, 180, 60),
               base_hues=(100, 280, 220, 340, 160, 40),
               hue_step_size=30,
               sl_values=[(80, 50), (95, 85)],
               gradient=True):
    """
    Get nice colors for graphs, using HLS values as input.

    Arguments:
        base_hues: (0 to 360) Values of the major hues
        hue_step_size: (1, 360) Values of the increases for the minor hues in the graph. Smaller steps will give more colors, but make it harder to see differences.
        saturation_values: (list of int from 0 to 100) List of values for saturation to use, on top of the several hues.
        lightness_values: (list of int from 0 to 100) List of values for lightness to use, on top of the several hues and saturation values.
        gradient: (bool) If True, organize colors by hues, else, cycle over hues before changing lightness/saturation
    """
    # Get the hues variants
    max_n_hues = int(np.diff(sorted(base_hues)).mean() / hue_step_size)
    n_hues = min(int(ceil(n / len(base_hues))), max_n_hues)
    hue_steps = range(0, hue_step_size * n_hues, hue_step_size)
    if gradient == True:
        hues = [h + i for h in base_hues for i in hue_steps]
    else:
        hues = [h + i for i in hue_steps for h in base_hues]
    n_sl = min(int(ceil(n / len(hues))), len(sl_values))

    if gradient == True:
        colors = [hls_to_rgb(h / 360, l / 100, s / 100) for h in hues for s, l in sl_values[:n_sl]]
    else:
        colors = [hls_to_rgb(h / 360, l / 100, s / 100) for s, l in sl_values[:n_sl] for h in hues]

    return colors[:n]


def shades(rgb, n, gradient=False, r=(0.1, 0.9)):
    """
    Give evenly spread lightness variants of a given color.

    Arguments:
        rgb: RGB color (3tuple of int[0, 255] or float[0, 1])
        n: number of shades to return (int)
        gradient: sort by shade value (bool)
        r: the range of values to pick from (2tuple of float[0, 1])

    Returns:
        list of RGB tuples
    """
    if type(rgb[0]) is int:
        rgb = [i / 255 for i in rgb]
    h, l, s = rgb_to_hls(*rgb)
    step = (r[1] - r[0]) / n
    variants = list()
    for i in range(n):
        v = l + i * step
        if v > r[1]:
            v -= r[1]
        if v < l:
            v += r[0]
        variants.append(v)
    if gradient == True:
        variants = sorted(variants)
    return [hls_to_rgb(h, v, s) for v in variants]


def tones(rgb, n, gradient=False, r=(0.1, 1)):
    """
    Give evenly spread saturation variants of a given color.

    Arguments:
        rgb: RGB color (3tuple of int[0, 255] or float[0, 1])
        n: number of tones to return (int)
        gradient: sort by tone value (bool)
        r: the range of values to pick from (2tuple of float[0, 1])

    Returns:
        list of RGB tuples
    """
    if type(rgb[0]) is int:
        rgb = [i / 255 for i in rgb]
    h, l, s = rgb_to_hls(*rgb)
    step = (r[1] - r[0]) / n
    variants = list()
    for i in range(n):
        v = s + i * step
        if v > r[1]:
            v -= r[1]
        if v < s:
            v += r[0]
        variants.append(v)
    if gradient == True:
        variants = sorted(variants)
    return [hls_to_rgb(h, l, v) for v in variants]


def nuances(rgb, n, gradient=False, r=(0, 1)):
    """
    Give evenly spread hue variants of a given color.

    Arguments:
        rgb: RGB color (3tuple of int[0, 255] or float[0, 1])
        n: number of nuances to return (int)
        gradient: sort by nuance value (bool)
        r: the range of values to pick from (2tuple of float[0, 1])

    Returns:
        list of RGB tuples
    """
    if type(rgb[0]) is int:
        rgb = [i / 255 for i in rgb]
    h, l, s = rgb_to_hls(*rgb)
    step = (r[1] - r[0]) / n
    variants = list()
    for i in range(n):
        v = h + i * step
        if v > r[1]:
            v -= r[1]
        if v < h:
            v += r[0]
        variants.append(v)
    if gradient:
        variants = sorted(variants)
    return [hls_to_rgb(v, l, s) for v in variants]


def blend_rgb_colors(*colors):
    """Blend RGB colors together.

    Parameters:
    ----------
    colors: 3-tuples
        All the RGB colors to blend, with matplotlib-style values 0 to 1.

    """
    c = list()
    for i in zip(*colors):
        c.append(min(np.sum(i), 1))
    return c
