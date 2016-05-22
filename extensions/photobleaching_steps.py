# -*- coding: utf8 -*-
"""Find the number of photobleaching steps for a given particle."""
from __future__ import absolute_import, division, print_function
from matplotlib import pyplot as plt
from ..utils.steps import find_steps_lstsq2 as find_steps


def photobleaching(self):
    """Photobleaching."""
    steps = list()
    for p in self:
        t, i = p.intensity()
        steps.append(len(find_steps(i)))
    return steps
    # plt.hist(steps, bins=[0, 1, 2, 3, 4, 5])
    # plt.show()


__extension__ = {'Particles': {'photobleaching': photobleaching}}
