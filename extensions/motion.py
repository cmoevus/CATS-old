# -*- coding: utf8 -*-
"""Calculate the speed of a particle."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp

def speed(self, axis='x'):
    """Fit a line over a particle's plot of <axis> against time."""
    return np.polyfit(self.t, self[axis], 1)[0]


def acceleration(self, axis='x'):
    """Fit a line over a particle's plot of the derivative of <axis> against time."""
    return np.polyfit(self.t[:-1], np.diff(self[axis]), 1)[0]

__extension__ = {
    'Particle': {
        'speed': speed,
        'acceleration': acceleration
    }
}
