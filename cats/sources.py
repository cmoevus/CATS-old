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
from . import extensions


@extensions.append
class Images(object):
    #
    # Make sure to update Experiment's sources/datasets function when this inherits from AttributeDict, to deal with the __iter__...
    #

    """
    Access raw images and metadata.

    Images are what comes out of an experiment. It is the raw, unanalysed,
    unfiltered, untransformed set of data recorded by the camera.

    Keyword argument:
        source: the path to the file or directory of images. In the latter case, the path can contain wildcards to select specific files within the dict.
    """

    def __init__(self, source):
        """Load either BioFormats or Tiff list."""
        self.dormant = True
        self.source = source
        self._dims = None
        self._len = None

    @property
    def source(self):
        """Return the path to the images."""
        return self._source

    @source.setter
    def source(self, source):
        """Select between a BioFormats reader or a image list reader."""
        source = globify(source)
        n_files = len(glob(source))
        if n_files == 0:
            raise ValueError('Source {0} is empty or does not exist.'.format(source))
        elif n_files == 1:
            self.is_file = True
            javabridge.start_vm(class_path=bioformats.JARS)
            self.reader = bioformats.ImageReader(source)
        else:
            self.is_file = False
        self._source = source
        self.dormant = False

    def check_source(self):
        """
        Activate the source and return True if source can be read, raise error if not.

        When loading (unpickling) an Images object, if the source is not accessible, the object's properties still will be. However, one may want to plug the source back in on the fly. If the source suddenly becomes accesible, this function will 'activate' the object for reading images.
        """
        if len(glob(self.source)) == 0:
            self.dormant = True
            raise ValueError('Source {0} is not accesible.'.format(self.source))
        else:
            if self.dormant == True:
                self.source = self.source
                self.dormant = False
            return True

    @property
    def dimensions(self):
        """Return the shape of the images in the format (x, y), in pixels."""
        if self._dims is None:
            self.check_source()
            self._dims = io.imread(glob(self.source)[0]).shape[::-1] if self.is_file == False else (self.reader.rdr.getSizeX(), self.reader.rdr.getSizeY())
        return self._dims

    @dimensions.setter
    def dimensions(self, value):
        """Return an error: dimensions is read-only."""
        return AttributeError("The 'dimensions' attribute is read-only")

    @property
    def length(self):
        """Return the number of frames in the experiment."""
        if self._len is None:
            self.check_source()
            self._len = len(glob(self.source)) if self.is_file == False else self.reader.rdr.getSizeT()
        return self._len

    @length.setter
    def length(self, value):
        """Return an error: length is read-only."""
        return AttributeError("The 'length' attribute is read-only")

    @property
    def shape(self):
        """Return numpy-style (y, x) dimensions of the images."""
        return self.dimensions[::-1]

    @shape.setter
    def shape(self, value):
        """Return an error: shape is read-only."""
        return AttributeError("The 'shape' attribute is read-only")

    def read(self):
        """Iterate over the images in the dir/file, return a numpy array representing the image."""
        self.check_source()
        if self.is_file is False:
            for f in sorted(glob(self.source)):
                yield io.imread(f)
        elif self.is_file == True:
            for t in range(self.length):
                yield self.reader.read(t=t, rescale=False)

    def get(self, frame):
        """
        Return a numpy array representing the image at the given frame.

        Keyword argument:
            frame: the frame to return
        """
        self.check_source()
        if self.is_file == False:
            return io.imread(sorted(glob(self.source))[frame])
        elif self.is_file == True:
            return self.reader.read(t=frame, rescale=False)

    def __repr__(self):
        """Return the path to the images."""
        return self.source

    def __str__(self):
        """Return the path to the images."""
        return self.source

    def _save_attributes(self):
        """Put the attributes from the Images into the object, in view of pickling and unpickling without the source."""
        try:
            self.dimensions
            self.length
        except:
            pass

    def __getstate__(self):
        """Pickle the whole object, except the reader."""
        # Put info in memory
        self._save_attributes()
        return dict(((k, v) for k, v in self.__dict__.items() if k != 'reader'))

    def __setstate__(self, state):
        """Load the whole object, but tolerate the absence of the source."""
        for k, v in state.items():
            if k != '_source':
                setattr(self, k, v)
        try:
            setattr(self, 'source', state['_source'])
        except:
            self._source = state['_source']
            self.dormant = True


@extensions.append
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
            """Build the image object and set the limits."""
            self.images = images if isinstance(images, Images) is True else Images(images)
            self.x, self.y, self.t, self.channel = x, y, t, c

    def _slicify(self, s, fallback=slice(None)):
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
            if s[1] == None:
                s[1] = fallback
            s = slice(*s)
        elif isinstance(s, int):
            s = slice(0, s)
        else:
            s = self._slicify(fallback)
        return s

    @property
    def source(self):
        """Return the path to images."""
        return self.images.source

    @source.setter
    def source(self, source):
        """Return an error: source of an ROI is read-only."""
        raise AttributeError("The source of a ROI is read-only. Directly change the source from the original Images object.")

    @property
    def dimensions(self):
        """Return the dimensions of the ROI, as (x, y)."""
        return self.x.stop - self.x.start, self.y.stop - self.y.start

    @property
    def length(self):
        """Return the length of the ROI, in frames."""
        return self.t.stop - self.t.start

    @property
    def x(self):
        """Return the limits of the ROI in x."""
        return self._x

    @x.setter
    def x(self, value):
        """Set the limits of the ROI in x."""
        self._x = self._slicify(value, self.images.dimensions[0])

    @property
    def y(self):
        """Return the limits of the ROI in y."""
        return self._y

    @y.setter
    def y(self, value):
        """Set the limits of the ROI in y."""
        self._y = self._slicify(value, self.images.dimensions[1])

    @property
    def t(self):
        """Return the limits of the ROI in time."""
        return self._t

    @t.setter
    def t(self, value):
        """Set the limits of the ROI in time."""
        self._t = self._slicify(value, self.images.length)

    def read(self):
        """Return a generator of ndarrays of every images in the source."""
        self.images.check_source()
        if self.images.is_file is False:
            for f in sorted(glob(self.images.source))[self.t]:
                yield io.imread(f)[self.y, self.x]
        elif self.images.is_file == True:
            for t in range(self.t.start, self.t.stop):
                yield self.images.reader.read(t=t, c=self.channel, rescale=False)[self.y, self.x]
        else:
            raise ValueError('Source is not accessible')

    def get(self, frame):
        """Return the given frame."""
        i = self.images.get(frame + self.t.start)[self.y, self.x]
        return i if i.ndim == 2 else i[:, :, self.channel]

    def __repr__(self):
        """Return the path to the images with the limits."""
        if self.images.is_file == False:
            r = {'source': self.source, 'x': self.x, 'y': self.y, 't': self.t}.__repr__()
        else:
            r = {'source': self.source, 'x': self.x, 'y': self.y, 't': self.t, 'channel': self.channel}.__repr__()
        return r

    def __str__(self):
        """Return the path to the images."""
        return self.source

    def __getstate__(self):
        """Return state. Do not inherit Images.__getstate__."""
        return self.__dict__

    def __setstate__(self, state):
        """Set state. Do not inherit Images.__setstate__."""
        for k, v in state.iteritems():
            setattr(self, k, v)


def globify(path):
    """Return a glob()-able path from the given path (add the '*' wildcard)."""
    if os.path.isdir(path):
        path = path + '*' if path[-1] == '/' else path + '/*'
    return path
