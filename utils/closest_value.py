# -*- coding: utf8 -*-
"""Find the closest numeric value in a given array."""


def get_closest_value(i, a):
    """
    Find the closest value in a given array.

    Arguments:
        i: value to match
        a: the array.
    """
    value = None
    distance = None
    for k in a:
        v = (k - i)**2
        if distance is None:
            value, distance = k, v
        elif v < distance:
                value, distance = k, v
    return value
