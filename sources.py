# -*- coding: utf8 -*-
"""
Handle images from different types of sources.

Images: access raw images from different sources.
ROI: access a specific region, in x, y, t, of a set of images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from glob import glob
# import javabridge
# import bioformats
import pims
import os
from . import extensions
from .utils.slicify import slicify
__all__ = ['ROI', 'globify']


def pil_mode_to_depth(v):
    """Transform the value v of PIL's image 'mode' into the depth of the image, in bits."""
    convert = {
        1: 1,
        "L": 8,
        "P": 8,
        "RGB": 8,
        "RGBA": 8,
        "CMYK": 8,
        "YCbCr": 8,
        "I": 32,
        "I;16": 16,
        "F": 32
    }
    return convert(v)


def bioformats_pixeltype_to_depth(v):
    """Transform the value v of bioformats' getPixelType() into the depth of the image, in bits."""
    convert = {
        7: 1,  # DOUBLE, not sure what this is
        6: 1,  # FLOAT, not sure what this is
        2: 16,  # INT16
        4: 32,  # INT32
        0: 8,  # INT8
        3: 16,  # UINT16
        5: 32,  # UINT32
        1: 8,  # UINT8
    }
    return convert(v)


@extensions.append
class ROI(object):

    """
    Select a specific region in x, y, t, c from a given Images object.

    ROI adds limits to Images object. It is a filtered Images object. It behaves entirely like an Images object, plus the filtering.

    Keyword arguments:
        images: the source Images object, or a path to the images (see doc from Images).
        x <slice, tuple, int>: the initial and final (excluded) pixel to select, in the x axis. If None, selects all pixels. If int, only one pixel.
        y <slice, tuple, int>: slice (or list) of the initial and final (excluded) pixel to select, in the y axis. If None, selects all pixels. If int, only one pixel.
        t <slice, tuple, int>: slice (or list) of the initial and final (excluded) frame to select. If None, selects all frames. If int, only one frame.

    """

    def __init__(self, images, x=None, y=None, t=None):
            """Build the image object and set the limits."""
            self.images = pims.open(images) if type(images) == str else images
            self.x, self.y, self.t = x, y, t

    @property
    def path(self):
        """Return the path to the images."""
        try:
            self._path = self.images.__dict__['_ancestor'].__dict__['pathname']
        except KeyError:
            self._path = self.images.__dict__['pathname']
        return self._path

    @path.setter
    def path(self, value):
        raise AttributeError("The path of a ROI is read-only. Directly change the source from the original Images object.")

    @property
    def dimensions(self):
        """Return the dimensions of the ROI, as (x, y)."""
        return abs(self.x.stop - self.x.start), abs(self.y.stop - self.y.start)

    @property
    def shape(self):
        """Return the dimensions of the ROI, as (x, y)."""
        return self.dimensions[::-1]

    @property
    def length(self):
        """Return the length of the ROI, in frames."""
        return self.t.stop - self.t.start

    @property
    def x(self):
        """Return the limits of the ROI in x, from the source object."""
        return self._x

    @x.setter
    def x(self, value):
        """Set the limits of the ROI in x."""
        self._x = slicify(value, slice(0, self.images[0].shape[1], 1))

    @property
    def y(self):
        """Return the limits of the ROI in y."""
        return self._y

    @y.setter
    def y(self, value):
        """Set the limits of the ROI in y."""
        self._y = slicify(value, slice(0, self.images[0].shape[0], 1))

    @property
    def t(self):
        """Return the limits of the ROI in time."""
        return self._t

    @t.setter
    def t(self, value):
        """Set the limits of the ROI in time."""
        self._t = slicify(value, slice(0, len(self.images), 1))

    def read(self):
        """Return a generator of ndarrays of every images in the source."""
        for t in range(self.t.start, self.t.stop):
            i = self.images[t][self.y, self.x]
            yield i

    def get(self, frame):
        """Return the given frame."""
        f = self.images[frame * self.t.step + self.t.start][self.y, self.x].copy()
        f.frame_no = frame
        return f
        # return self.images[frame * self.t.step + self.t.start][self.y, self.x].view(np.ndarray)  # This sucks

    def __getitem__(self, key):
        """Implement indexing."""
        return self.get(key)

    # def __repr__(self):
    #     """Return the path to the images with the limits."""
    #     r = 'ROI of {0} with '.format(self.path)
    #     for n, l in zip(['x', 'y', 't'], ['abs_x', 'abs_y', 'abs_t']):
    #         v = getattr(self, l)
    #         r += '{0} = ({1}, {2}), '.format(n, v.start, v.stop)
    #     c = [self.c] if type(self.c) is int else self.c
    #     r = r[:-2] + ', channel{0}'.format('s ' if len(c) > 1 else ' ')
    #     for i in c:
    #         r += str(i) + ', '
    #     return r[:-2]

    def __len__(self):
        """Get a len() output."""
        return self.length

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.length:
            self.counter += 1
            return self.get(self.counter - 1)
        else:
            raise StopIteration




def globify(path):
    """Return a glob()-able path from the given path (add the '*' wildcard)."""
    if os.path.isdir(path):
        path = path + '*' if path[-1] == '/' else path + '/*'
    return path


def same_images(a, b):
    """
    Check if two source objects (Images, ROI) refer to the same source of images.

    Arguments:
        a, b: the objects to compare
    """
    return a.source.path == b.source.path


def same_time(a, b):
    """
    Check if two source objects (Images, ROI) coincide in absolute time.

    Arguments:
        a, b: the objects to compare
    """
    if isinstance(a, ROI):
        t_a = a.abs_t
    else:
        t_a = slice(0, a.length, 1)

    if isinstance(b, ROI):
        t_b = a.abs_t
    else:
        t_b = slice(0, b.length, 1)

    return t_a == t_b


def share_channels(a, b):
    """
    Check if two source objects (Images, ROI) refer to the same source of images and if they have the same channel, or, minimally, if one set of channels encompass the other set.

    Arguments:
        a, b: the objects to compare
    """
    if same_images(a, b):
        sets = list()
        if type(a) == ROI:
            sets.append(a.c)
        if type(b) == ROI:
            sets.append(b.c)
        # A. One Images object, at least. Images contain all channels.
        if len(sets) < 2:
            return True
        # B. Two ROI objects
        else:
            return set(sets[0]).issubset(sets[1]) or set(sets[0]).issuperset(sets[1])
    else:
        return False


def same_channel(a, b):
    """
    Check if two source objects (Images, ROI) refer to the same source of images and if they have the same channel, or, minimally, if one set of channels encompass the other set.

    Arguments:
        a, b: the objects to compare
    """
    if same_images(a, b):
        if type(a) == ROI and type(b) == ROI:
            return a.c == b.c
        elif type(a) == Images and type(b) == Images:
            return True
        else:
            return False
    else:
        return False


def same_roi(a, b):
    """
    Check if two source objects refer to the same source of images and have the same limits.

    Arguments:
        a, b: the objects to compare
    """
    if same_images(a, b) and isinstance(a, ROI) and isinstance(b, ROI) and a.x == b.x and a.y == b.y and a.t == b.t and a.c == b.c:
        return True
    else:
        return False
