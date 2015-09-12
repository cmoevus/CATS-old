#!env /usr/bin/python
# -*- coding: utf8 -*-
"""
Handle images from different types of sources.

Images: access raw images from different sources.
ROI: access a specific region, in x, y, t, of a set of images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from glob import glob
import javabridge
import bioformats
import os
from skimage import io


class Images(object):

    """
    Access raw images and metadata.

    Images are what comes out of an experiment. It is the raw, unanalysed,
    unfiltered, untransformed set of data recorded by the camera.

    Keyword argument:
        source: the path to the file or directory of images. In the latter case, the path can contain wildcards to select specific files within the dict.
    """

    def __init__(self, source):
        """ Load either BioFormats or Tiff list """
        # A file was given
        if os.path.isfile(source) is True:
            self.is_file = True
            self.source = source
            javabridge.start_vm(class_path=bioformats.JARS)
            self.reader = bioformats.ImageReader(self.source)

        # A path or dir was given
        else:
            if os.path.isdir(source) is True:
                wildcards = '*'
            else:
                source, wildcards = os.path.split(source)
                if os.path.isdir(source) is False:
                    raise ValueError('This path does not exists')
            self.is_file = False
            self.source = source + wildcards if source[-1] == '/' else source + '/' + wildcards

        # Initiate properties
        self._dims = None
        self._len = None

    @property
    def dimensions(self):
        """Return the shape of the images in the format (x, y), in pixels."""
        if  self._dims is None:
            self._dims = io.imread(glob(self.source)[0]).shape[::-1] if self.is_file == False else  self.reader.rdr.getSizeX(), self.reader.rdr.getSizeY()
        return self._dims

    @dimensions.setter
    def dimensions(self, value):
        """Return an error: dimensions is read-only."""
        return AttributeError("The 'dimensions' attribute is read-only")

    @property
    def length(self):
        """Return the number of frames in the experiment"""
        if  self._len is None:
            self._len =  len(glob(self.source)) if self.is_file == False else  self.reader.rdr.getSizeT()
        return self._len

    @length.setter
    def length(self, value):
        """Return an error: length is read-only."""
        return AttributeError("The 'length' attribute is read-only")

    def read(self):
        """Iterate over the images in the dir/file, return a numpy array representing the image."""
        if self.is_file is False:
            for f in sorted(glob(self.source)):
                yield io.imread(f)
        else:
            for t in range(self.length):
                yield self.reader.read(t=t, rescale=False)

    def get(self, frame):
        """
        Return a numpy array representing the image at the given frame

        Keyword argument:
            frame: the frame to return
        """
        if self.is_file == False:
            return io.imread(sorted(glob(self.source))[frame])
        else:
            return self.reader.read(t=frame, rescale=False)


class ROI(Images):
    """
    Select specific regions in x, y, t from a given Images object.

    ROI filters an Images object by adding limits to to it. It behaves entirely
     like an Images object, appart from this filtering.

    Keyword arguments:
        images: the source Images object, or a path to the images (see doc from Images).
        x: slice (or list) of the initial and final (excluded) pixel to select, in the x axis.
        y: slice (or list) of the initial and final (excluded) pixel to select, in the y axis.
        t: slice (or list) of the initial and final (excluded) frame to select.
        c: channel to use (indice of the first channel is 0). Only applies to image files. Not dirs.
    """

    def __init__(self, images, x=None, y=None, t=None, c=0):
            """ Build the image object and set the limits"""
            self.images = images if isinstance(images, Images) is True else Images(images)
            self.x, self.y, self.t , self.channel = x, y, t, c

    def _slicify(self, s, fallback = slice(None)):
        """
        Transform the input into a slice.

        Acceptable input:
            slice
            list/tuple
            int: the upper limit
            Everything else will be replaced by a _slicified given fallback value
        """
        if isinstance(s, slice):
            pass
        elif hasattr(s, '__iter__'):
            s = slice(*s)
        elif isinstance(s, int):
            s = slice(0, s)
        else:
            s = self._slicify(fallback)
        return s

    @property
    def dimensions(self):
        return self.x.stop - self.x.start, self.y.stop - self.y.start

    @property
    def length(self):
        return self.t.stop - self.t.start

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = self._slicify(value, self.images.dimensions[0])

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = self._slicify(value, self.images.dimensions[1])

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = self._slicify(value, self.images.length)

    def read(self):
        if self.images.is_file is False:
            for f in sorted(glob(self.images.source))[self.t]:
                yield io.imread(f)[self.y, self.x]
        else:
            for t in range(self.t.start, self.t.stop):
                yield self.images.reader.read(t=t, c=self.channel, rescale=False)[self.y, self.x]

    def get(self, frame):
        return self.images.get(frame + self.t.start)[self.y, self.x, self.channel]