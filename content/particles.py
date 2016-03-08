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

from __future__ import absolute_import, division, print_function
from .. import extensions
from .abstract import contents, content
__all__ = ['Particles', 'Particle']


@extensions.append
class Particles(contents):

    """
    List of particles detected from different sources.

    Inherits from cats.content.Content.
    """

    def __init__(self, *args, **kwargs):
        """Start the object and define the base units."""
        super(Particles, self).__init__(*args, **kwargs)
        # self._units = {'x': 'px',
        #                'y': 'px',
        #                'sx': 'px',
        #                'sy': 'px',
        #                'i': 'AU',
        #                't': 'f'} if '_units' not in kwargs else kwargs['_units']


@extensions.append
class Particle(content):

    """
    Single particle represented as a numpy recarray.

    Minimal columns and names of the array:
        x: (float) position in x (pixels)
        y: (float) position in y (pixels)
        t: (int) frame of the detection (spot)
    Additional columns may be given by the processor.
    """
