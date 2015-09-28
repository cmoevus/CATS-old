#!env /usr/bin/python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
from multiprocessing import cpu_count
from .adict import adict
__all__ = ['Experiment', 'Images']

Experiment = adict({
    'max_processes': cpu_count(),
    'particles': {
        'find': 'stable_particles',  # The method to use from the linkage module
        'max_disp': (2, 2),  # Maximum displacement, in pixels
        'max_blink': 25,  # Maximum blinking, in frames
        'keep_unfit': True,   # Keep the spots even if a Gaussian could not be fitted on them (no sub-pixel resolution)
        'min_length': 3,  # Minimum number of frames to keep a track
        'max_length': None,   # The maximum number of frames to keep a track
        'mean_blink': 5,  # The mean number of frames between two spots to keep a track
    },
    'barriers': {
        # Values for Charlie Brown DT: dbp = 12.5um and dbb = 37um
        'dbp': 47,  # Distance between the barriers and the pedestals in pixels
        'dbb': 139,  # Distance between two sets of barriers, in pixels
        'axis': 'y',  # The axis of the image with which the barriers are parallel. Either 'x' or 'y'
        'orientation': 'bp',  # The order of appearance of barriers and pedestals, from the left. bp: barrier, then pedestals. pb: pedestals, then barriers
    }
})

Images = adict({
    'barrier_detection': {
        'frames': None,  # The frames to use for finding the barriers. If None, uses all the images.
    },
    'barrier_detection': {
        'approx': 20,  # Approximate
        'blur': 2,  # Blurring for diminishing the effect of noise.
        'guess': True,  # Guess the position of barriers when there
                        # expected but not present
        'tlims': None
    },
})
