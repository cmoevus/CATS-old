# -*- coding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .content import ContentUnit


class Noise(ContentUnit):
    """
    Description of the noise in a source.

    Inherits from ContentUnit

    Minimal fields are:
        t: (int) the frame of the recorded values
        m: (float) the mean of the Gaussian
        s: (float) the standard deviation of the Gaussian
    """
