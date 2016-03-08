# -*- coding: utf8 -*-
"""
Filaments are the DNA and other long, rod-shaped signal in images.

How to write processors for filaments:
======================================
These directives (on top of the other general Howtos in the doc from cats.processors) will make your processor usable:
    - Return a "Filaments" object
    - Set the processor function in Filaments.processor
    - Set the image source in Filaments.sources
    - Set the values of the arguments used by the processor to find the filaments as attributes of the Filaments object.
    - Populate the "Filaments object" with "Filament" objects (see the doc of the class for more info).
    - The attribute 'source' of each Filament returns its source object (the one given to the processor)
"""
from __future__ import absolute_import, division, print_function
from .. import extensions
from .abstract import contents, content
__all__ = ['Filaments', 'Filament']


@extensions.append
class Filaments(contents):

    """
    List of filaments from different sources

    Inherits from cats.content.Content.
    """


@extensions.append
class Filament(content):

    """
    Single filament represented as a numpy recarray.

    Minimal columns and names of the array:
        x: (float) initial pixel, in x
        y: (float) initial pixel, in y
        l: (float) length of the filament (pixels)
        a: (float) angle between the filament and the x axis, in radians.
        t: (int) frame of the detection (spot)
    Additional columns may be given by the processor.
    """
