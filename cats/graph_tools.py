# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from colorsys import hls_to_rgb
from math import ceil


def get_colors(n=72,
               base_hues=(120, 300, 240, 0, 180, 60),
               hue_step_size=30,
               sl_values=[(75, 60), (100, 90)],
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
    # Starting ranges

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
