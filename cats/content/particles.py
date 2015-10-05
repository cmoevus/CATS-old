# -*- coding: utf8 -*-
"""
Particles are the proteins and other globular, Gaussian-shaped signal in images.

How to write processors for particles:
======================================
These directives (on top of the other general Howtos in the doc from cats.processors) will make your processor usable:
    - Return a "Particles" object
    - Set the processor function in Particles.processor
    - Set the image source in Particles.sources
    - Set the values of the arguments used by the processor to find the particles in the object
    - Populate the "Particles object" with "Particle" objects (see the doc of the class for more info).
    - The attribute 'source' of each Particle returns its source object (the one given to the processor)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .. import extensions
from ..content import Content, ContentUnit
__all__ = ['Particles', 'Particle']


@extensions.append
class Particles(Content):

    """
    List of particles detected from different sources.

    Inherits from cats.content.Content.
    """


@extensions.append
class Particle(ContentUnit):

    """
    Single particle represented as a numpy recarray.

    Minimal columns and names of the array:
        x: (float) position in x (pixels)
        y: (float) position in y (pixels)
        t: (int) frame of the detection (spot)
    Additional columns may be given by the processor.
    """
