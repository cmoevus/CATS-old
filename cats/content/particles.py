"""
Particles are the proteins and other globular, Gaussian-shaped signal in images.

How to write processors for particles:
====================================
These directives (on top of the other general Howtos in the doc from cats.processors) will make your processor usable:
    - Write the list of particles as follow:
        - Each particle is a numpy recarray.
        - Each line in the recarray is one spot (call it 'detection' or 'gaussian'. It is what is on one frame).
        - Minimally, each particle must have the following columns:
            - 'x': (numerical) the position in x
            - 'y': (numerical) the position in y
            - 't': (int) the frame number
        - You can add as many fields as you want. The order does not matter, and this is the interest of the recarray versus lists. Here are a few standardized column names for particles:
            - 'i': intensity at the center of the spot
            - 's': the sigma of the gaussian of the spot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from .. import extensions
from ..content import Content
__all__ = ['Particles', 'Particle']


@extensions.append
class Particles(Content):

    """
    List of particles detected from different sources.

    Object has attribute sources which is a list of sources (can be shared with Experiment, for example, or other types of content)

    """


@extensions.append
class Particle(np.recarray):

    """
    Single particle represented by a numpy recarray.
    Minimal shape and name of the array:
        x: (float) position in x (pixels)
        y: (float) position in y (pixels)
        t: (int) frame of the detection (spot)
    Proposed additional fields:
        s: (float) sigma of the Gaussian of the detection
        a: (float) amplitude of the Gaussian of the detection (intensity at peak)

    Supports extensions like lifetime(), diffusion_coefficient(), etc.
    """

    def __init__(self, *args, **kwargs):
        """Initiate the recarray."""
        super(Particle, self).__init__(args)

    def __reduce__(self):
        """Prepare the object for pickling. Inherit from the Numpy function and add the object's dictionary to the state."""
        func, args, state = np.recarray.__reduce__(self)
        state += (self.__dict__, )
        return (func, args, state)

    def __setstate__(self, value):
        """Unpickle the object's state. Apply the state of the Numpy object and add the dictionary of the object. """
        super(Particle, self).__setstate__(value[0:5])
        self.__dict__ = value[5]
