from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def max_intensity(self):
    """Find the maximum intensity across all images."""
    max_i = 0
    for i in self.read():
        max_i_i = i.max()
        if i.max() > max_i:
            max_i = max_i_i
    return max_i


def min_intensity(self):
    """Find the minimum intensity across all images."""
    min_i = 0
    for i in self.read():
        min_i_i = i.min()
        if i.min() > min_i:
            min_i = min_i_i
    return min_i


__extension__ = {'ROI': {
    'max_intensity': max_intensity,
    'min_intensity': min_intensity,
}}
