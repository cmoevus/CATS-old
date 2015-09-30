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
import numpy as np
from glob import glob
import javabridge
import bioformats
import os
from skimage import io
from . import extensions
__all__ = ['ImageDir', 'Images', 'ROI', 'globify']


class ImageDir(object):

    """
    Read a directory of images, or a list of directories or images.

    Each given path (as understood by Python's glob module) is considered a channel.
    This class aims to behave like bioformats.ImageReader, to the eyes of the Images class. Hence the weird stuff. It was not meant to be beautiful.

    You are in charge of making sure that all sources have the same length and the same dimensions. This will not do it for you. Beware the weird errors.

    Arguments:
        Paths (including wildcards) to the directories of interest. Each path is one channel. You can name channels using keyword arguments. The channels will be numbered in the order you gave them.
    """

    def __init__(self, *args, **kwargs):
        """Set the sources as channels."""
        channels, name = dict(), 0
        for source in args:
            if type(source) is dict:
                channels.update(source)
            else:
                channels[str(name)] = source
                name += 1
        channels.update(kwargs)
        self.channels, self.sources = list(), list()
        for name, source in channels.items():
            source = globify(source)
            n_files = len(glob(source))
            if n_files == 0:
                raise ValueError('Source {0} is empty or does not exist.'.format(source))
            else:
                self.channels.append(name)
                self.sources.append(source)
        # Compatibility with bioformats.ImageReader
        self.rdr = self

    def getSizeX(self):
        """Return the size in the x dimension (in pixels)."""
        return self.read(c=0, t=0, rescale=False).shape[1]

    def getSizeY(self):
        """Return the size in the y dimension (in pixels)."""
        return self.read(c=0, t=0, rescale=False).shape[0]

    def getImageCount(self):
        """Return the number of images in series."""
        return self.getSizeT()

    def getSizeT(self):
        """Return the number of images in series."""
        length = len(glob(self.sources[0]))
        if length == 0:
            raise ValueError('Source {0} is empty or does not exist.'.format(self.sources[0]))
        else:
            return length

    def getSizeC(self):
        """Return the number of channels."""
        return len(self.sources)

    def getSizeZ(self):
        """Return 1. ImageDir does not support z stacks."""
        return 1

    @property
    def path(self):
        """Return a dict that could be used to initiate the object."""
        if len(self.channels) > 1 or self.channels[0] != '0':
            return dict(zip(self.channels, self.sources))
        else:
            return self.sources[0]

    @path.setter
    def path(self, value):
        """Prevent wrongdoers from using path."""
        raise ValueError('"path" is read-only')

    def read(self, c=None, t=0, rescale=True):
        """Return an image from the sources."""
        # A. Fetch the image(s)
        if c is None:
            c = self.channels
        elif '__iter__' not in dir(c):
            c = [c]
        image = list()
        for channel in c:
            try:
                if type(channel) is int:
                    source = self.sources[channel]
                else:
                    source = self.sources[self.channels.index(channel)]
            except (IndexError, ValueError):
                raise ValueError('Channel "{0}" does not exist'.format(channel))
            try:
                img = io.imread(glob(source)[t])
                if rescale == True:  # This is stupid because it is relative to this image, not the series. This will change the intensity scaling factor between images. But I did not choose.
                    img = img.astype(float) / img.max()
                image.append(img)
            except IndexError:
                if os.path.exists(source) == False:
                    e = ['Source', source]
                else:
                    e = ['Frame', t]
                raise ValueError('{0} {1} does not exist'.format(*e))

        # B. Make a 3D array if need be.
        if len(image) > 1:
            img = np.zeros((image[0].shape[0], image[0].shape[1], len(image)), dtype=img.dtype)
            for n, i in enumerate(image):
                img[:, :, n] = image[n]
            image = img
        else:
            image = image[0]

        return image

    def __repr__(self):
        """Funkyprint the sources."""
        if len(self.sources) > 1:
            base = os.path.split(os.path.commonprefix(self.sources))[0] + '/'
            ext = ''
            for source in self.sources:
                ext += source.replace(base, '') + ', '
            return base + '[' + ext[:-2] + ']'
        else:
            return self.sources[0]


@extensions.append
class Images(object):
    #
    # Make sure to update Experiment's sources/datasets function when this inherits from AttributeDict, to deal with the __iter__...
    #

    """
    Access raw images.

    Images are what comes out of an experiment. It is the raw, unanalysed,
    unfiltered, untransformed set of data recorded by the camera.

    This object will save information about the images so that it can be used even in the absence of the images, post pickling/unpickling. It does not save the images, only basic metadata about them.

    Keyword argument:
        source: the path to the file or directory of images. In the latter case, the path can contain wildcards to select specific files within the dict.
    """

    def __init__(self, source):
        """Load either BioFormats or ImageDir image readers."""
        self.source = source

    @property
    def source(self):
        """Return the path to the images."""
        if '_source' in self.__dict__:
            return self._source
        else:
            raise ValueError("The source '{0}' is not accessible.".format(self.path))

    @source.setter
    def source(self, source):
        """Set the appropriate source reader."""
        # A. Path: bioformats or ImageDir?
        if type(source) is str:
            source = globify(source)
            n_files = len(glob(source))
            if n_files == 0:
                raise ValueError('Source {0} is empty or does not exist.'.format(source))
            elif n_files == 1:
                javabridge.start_vm(class_path=bioformats.JARS)
                self._source = bioformats.ImageReader(source)
            else:
                self._source = ImageDir(source)
        # B. List or dict: ImageDir
        elif type(source) in [tuple, set, list]:
            self._source = ImageDir(*source)
        elif type(source) is dict:
            self._source = ImageDir(**source)
        # C. Object
        elif isinstance(source, bioformats.ImageReader) or isinstance(source, ImageDir):
            self._source = source
        self._save_attributes()

    def activate(self):
        """
        Activate the source and return True if source can be read, raise error if not.

        When loading (unpickling) an Images object, if the source is not accessible, the object's properties still will be. However, one may want to plug the source back in on the fly. If the source suddenly becomes accesible, this function will 'activate' the object for reading images.
        """
        if '_source' in self.__dict__:
            return True
        else:
            try:
                self.source = self.path
            except:
                raise ValueError('Source is not accessible.')

    @property
    def dimensions(self):
        """Return the shape of the images in the format (x, y), in pixels."""
        try:
            self._dims = self.source.rdr.getSizeX(), self.source.rdr.getSizeY()
        except:
            pass
        try:
            return self._dims
        except:
            raise ValueError('Source files cannot be read.')

    @dimensions.setter
    def dimensions(self, value):
        """Return an error: dimensions is read-only."""
        return AttributeError("The 'dimensions' attribute is read-only")

    @property
    def length(self):
        """Return the number of frames in the experiment."""
        try:
            # BioFormats considers Tiff stacks to be in Z. We consider them to be in T. ImageCount is independent of T or Z.
            self._len = self.source.rdr.getImageCount()
        except:
            pass
        try:
            return self._len
        except:
            raise ValueError('Source files cannot be read.')

    @length.setter
    def length(self, value):
        """Return an error: length is read-only."""
        return AttributeError("The 'length' attribute is read-only")

    @property
    def channels(self):
        """Return the number of channels."""
        try:
            self._channels = self.source.rdr.getSizeC()
        except:
            pass
        try:
            return self._channels
        except:
            raise ValueError('Source files cannot be read.')

    @property
    def path(self):
        """Return the path to the source files."""
        try:
            self._path = self.source.path
        except:
            pass
        try:
            return self._path
        except:
            raise ValueError('Source files became inaccessible before I could read them...')

    @path.setter
    def path(self, value):
        """Return an error: path is read-only."""
        return AttributeError("The 'path' attribute is read-only. Set the source of files with the 'source' attribute.")

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
        self.activate()
        if self.length == self.source.rdr.getSizeZ():
            # BioFormats considers Tiff stacks to be in Z. We consider them to be in T.
            for t in range(self.length):
                yield self.source.read(z=t, rescale=False)
        else:
            for t in range(self.length):
                yield self.source.read(t=t, rescale=False)

    def get(self, frame):
        """
        Return a numpy array representing the image at the given frame.

        Keyword argument:
            frame: the frame to return
        """
        self.activate()
        if self.length == self.source.rdr.getSizeZ():
            # BioFormats considers Tiff stacks to be in Z. We consider them to be in T.
            return self.source.read(z=frame, rescale=False)
        else:
            return self.source.read(t=frame, rescale=False)

    def __repr__(self):
        """Return the path of the images."""
        return 'Images from ' + self.source.__repr__()


    def _save_attributes(self):
        """Put the attributes from the Images into the object, in view of pickling and unpickling without the source."""
        for attr in [i for i in dir(self)]:
            try:
                getattr(self, attr)
            except:
                pass

    def __getstate__(self):
        """Pickle the whole object, except the reader."""
        # Put info in memory
        self._save_attributes()
        return dict(((k, v) for k, v in self.__dict__.items() if k != '_source'))

    def __setstate__(self, state):
        """Load the whole object, but tolerate the absence of the source."""
        for k, v in state.items():
            if k != '_source':
                setattr(self, k, v)
        try:
            setattr(self, 'source', state['_path'])
        except:
            pass


@extensions.append
class ROI(Images):

    """
    Select a specific region in x, y, t, c from a given Images object.

    ROI adds limits to Images object. It is a filtered Images object. It behaves entirely like an Images object, plus the filtering.

    Keyword arguments:
        images: the source Images object, or a path to the images (see doc from Images).
        x: slice (or list) of the initial and final (excluded) pixel to select, in the x axis. If None, selects all pixels.
        y: slice (or list) of the initial and final (excluded) pixel to select, in the y axis. If None, selects all pixels.
        t: slice (or list) of the initial and final (excluded) frame to select. If None, selects all frames.
        c: channels to use (indice of the first channel is 0). Only applies to image files. Not TIF dirs.  If None, selects all channels.
    """

    def __init__(self, images, x=None, y=None, t=None, c=None):
            """Build the image object and set the limits."""
            self.images = images if isinstance(images, Images) is True else Images(images)
            self.x, self.y, self.t, self.c = x, y, t, c

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
                s = (s[0], fallback)
            if s[0] == None:
                s = (0, s[1])
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

    @property
    def c(self):
        """Return the channels of the ROI."""
        return self._c

    @c.setter
    def c(self, value):
        """Set the list of channels to select."""
        value = range(self.images.channels) if value is None else value
        self._c = value

    def read(self):
        """Return a generator of ndarrays of every images in the source."""
        for t in range(self.t.start, self.t.stop):
            i = self.images.get(t)[self.y, self.x]
            yield i if i.ndim == 2 else i[:, :, self.c]

    def get(self, frame):
        """Return the given frame."""
        i = self.images.get(frame + self.t.start)[self.y, self.x]
        return i if i.ndim == 2 else i[:, :, self.c]

    def __repr__(self):
        """Return the path to the images with the limits."""
        r = 'ROI of {0} with limits:'.format(self.source)
        for l in ['x', 'y', 't']:
            v = getattr(self, l)
            r += '\n\t{0} from {1} to {2}, '.format(l, v.start, v.stop)
        c = [self.c] if type(self.c) is int else self.c
        r = r[:-1] + '\n\tchannel{0}'.format('s ' if len(c) > 1 else ' ')
        for i in c:
            r += str(i) + ', '
        return r[:-2]

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
