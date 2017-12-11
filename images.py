# -*- coding: utf8 -*-
"""Utility functions to help with image handling."""
from __future__ import absolute_import, division, print_function

import numpy as np
import colorsys
import warnings
import pims
import slicerator
import os
import skimage.io


class Images(object):
    """Import and access images folder, bioformats, etc.

    Import is based on PIMS and, as a consequence, all format supported are
    those supported by PIMS.

    Channels are not supported for now.

    The Images objects are indexable for frames and dimensions, in the order
    (frame, y, x). If only one frame is requested, it will return a
    pims.frame.Frame object. If several frames are requested, it will return an
    Images object sliced accordingly.

    Images objects can be pickled and unpickled. If they are unpickled in the
    absence of the source of the images, the user will, of course, not be able
    to access the images themselves, but will still be able to use the object.
    Notably, the 'shape' attribute will still be usable. If one brings the
    source back, one can use the 'reload' method to reload the object with
    the images.

    Examples of acceptable syntax:
    Images[0]  # Get the first frame
    Images[0, 5:10, 10:60]  # Get the first frame, between pixels 5 and 10 in y
                              and pixels 10 and 60 in x
    Images[:, :, :256]  # Get all images, truncated in x until pixel 256
    Images[::10]  # To get one out of ten images

    Parameters:
    -----------
    images: str, pims or cats.Images object
        The images to read. Can be a path, readable by `pims.open()`, or an object with images already loaded.
    frames: slice
        The limits on the frames to use. Frame 0 will be equal to the beginning of the slice.
    y: slice
        The limits on the pixels to use in the y axis.
    x: slice
        The limits on the pixels to use in the x axis.

    Example:
    --------
    >>> i = Images('/path/to/images/*.tif')
    >>> type(i[0])
    pims.frames.Frames
    >>> type(i[:10])
    cats.images.Images

    """

    def __init__(self, images, frames=slice(None), y=slice(None), x=slice(None)):
        """Instanciate the instance."""
        self.base = pims.open(images) if type(images) is str else images

        # Convert the limits to usable slices
        # Frames
        start = 0 if frames.start is None else frames.start
        stop = len(self.base) if frames.stop is None else frames.stop
        step = 1 if frames.step is None else frames.step
        frames = slice(start, stop, step)
        # x
        start = 0 if x.start is None else x.start
        stop = self.base[0].shape[1] if x.stop is None else x.stop
        step = 1 if x.step is None else x.step
        x = slice(start, stop, step)
        # y
        start = 0 if y.start is None else y.start
        stop = self.base[0].shape[0] if y.stop is None else y.stop
        step = 1 if y.step is None else y.step
        y = slice(start, stop, step)

        self.slices = [frames, y, x]

        # Store the path for future pickling and unpickling in a base-less context
        self.path

    def __getitem__(self, items):
        """Return the slice of or whole frame(s) as requested.

        Expects input as: frame, y, x

        """
        #
        # Implement advanced indexing!
        # For example, by checking the type of items. If list or tuple or slice, behave diferently. For exampe:
        # if list:
        #   return dataset[0][self.slices][items]  # First slice the image into the expected boundaries, then apply the advanced indexing.
        #

        # Fix the input
        if type(items) is not tuple:
            items = items,
        if len(items) < 3:
            items = list(items) + self.slices[len(items):]
        frame, y, x = items

        # A. Only return one frame
        if type(frame) is not slice:
            if frame < 0:
                frame = self.slices[0].stop + frame
            f = self.base[frame * self.slices[0].step + self.slices[0].start][y, x].copy()
            f.frame_no = frame
            return f

        # B. Return several frames as an Images object
        else:
            return Images(self, frames=frame, y=y, x=x)

    def __getattr__(self, attr):
        """Return the proper error when dealing with the absence of images, i.e. the absence of the "base" attribute, and with image related attributes."""
        if attr == 'base':
            try:
                self.reload()  # Check if the source magically reappeared first
                return self.base
            except (FileNotFoundError, pims.api.UnknownFormatError) as e:
                errormsg = 'Cannot access the images'
                if hasattr(self, '_path'):
                    errormsg += " at " + self._path
                raise FileNotFoundError(errormsg) from None
        else:
            try:
                return getattr(self.base[0], attr)  # Try to get the attribute from the images themselves
            except AttributeError:
                raise AttributeError("{0} object has no attribute {1}".format(self.__class__, attr))

    def __iter__(self):
        """Start iteration."""
        self._iter_counter = 0
        return self

    def __next__(self):
        """Continue iteration."""
        if self._iter_counter < len(self):
            self._iter_counter += 1
            return self.__getitem__(self._iter_counter - 1)
        else:
            raise StopIteration

    @property
    def path(self):
        """Return the path to the images."""
        if not hasattr(self, '_path'):
            t = type(self.base)
            if t is slicerator.Slicerator:
                self._path = self.base._ancestor.pathname  # Slicerator could do that job for us...
            elif t is Images:
                return self.base.path
            else:
                self._path = self.base.pathname  # Pims object
        return self._path

    @property
    def shape(self):
        """Return the shape of the frames as (y, x)."""
        return self.slices[1].stop - self.slices[1].start, self.slices[2].stop - self.slices[2].start

    def __len__(self):
        """Return the number of frames."""
        return self.slices[0].stop - self.slices[0].start

    def __getstate__(self):
        """Pickle the whole object, except the reader."""
        # Do not pickle source images.
        blacklist = []
        if 'pims' in str(type(self.base)):
            blacklist.append('base')
        return dict(((k, v) for k, v in self.__dict__.items() if k not in blacklist))

    def __setstate__(self, state):
        """Load the whole object, but tolerate the absence of the images."""
        for k, v in state.items():
            setattr(self, k, v)
        try:
            self.base  # Try to reload the pims images if need be
        except FileNotFoundError:
            pass

    def reload(self):
        """Reload the images in case the source was missing."""
        if 'base' in self.__dict__.keys() and type(self.base) is Images:
            self.base.reload()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.base = pims.open(self.path)

    def save(self, folder, prefix='', extension='tif'):
        """Save the images as a sequence into the given folder.

        Parameters:
        -----------
        folder: str
            the folder in which the save the data. If the folder does not exist, if will be created.
        prefix: format()able type
            the prefix to give to the files.
        extension: str
            the file extension. It has to be an extension writable by skimage.io.imsave

        """
        if not os.path.isdir(folder):
            os.makedirs(folder)
        nb = np.ceil(np.log10(len(self))) if len(self) > 0 else 0
        for i, image in enumerate(self):
            skimage.io.imsave(os.path.join(folder, "{}{:0{}d}.{}".format(prefix, i, int(nb), extension)), image)


def stack_images(*images, spacing=2, spacing_color=255, axis=0):
    """Stack images into one image.

    You need to provide 2D RGBs as input (height * width * 3). Sizes must match in the axis not being stacked.

    Parameters:
    ----------
    *images: np.array
        the images to stack
    spacing: int
        the number of pixels between each image
    spacing_color: int or 3-tuple
        the color, as number for grayscales or 3-tuple for RGB, of the pixels in between images
    axis: int
        the axis onto which to stack the images (0 for vertical, 1 for horizontal)

    Returns:
    -------
    image: np.array
        The stacked images

    """
    dims = np.array([i.shape[:2] for i in images])
    if axis == 0:
        width, height = dims[:, 1].max(), sum(dims[:, 0]) + (len(dims) - 1) * spacing
    else:
        height, width = dims[:, 0].max(), sum(dims[:, 1]) + (len(dims) - 1) * spacing
    stack = np.zeros((height, width, 3), dtype=np.uint8)
    stack.fill(spacing_color)
    position = 0
    for i in images:
        if axis == 0:
            stack[position: position + i.shape[0], :, :] = i
        else:
            stack[:, position: position + i.shape[1], :] = i
        position += i.shape[axis] + spacing
    return stack


def get_image_depth(image):
    """Return the pixel depth of the image (given as a ndarray from, for example, bioformats or scipy.misc.imread), in bits."""
    convert = {
        np.uint8: 8,
        np.uint16: 16,
        np.uint32: 32,
        np.uint64: 64,
        np.int8: 8,
        np.int16: 16,
        np.int32: 32,
        np.int64: 64,
    }
    try:
        return convert[image.dtype.type]
    except KeyError:
        raise ValueError('Unrecognized image type.')


def grayscale_to_rgb(grayscale):
    """Transform a grayscale image to a 8bits RGB.

    Parameters:
    ------------
        grayscale: numpy array
            the image to transform into 8bit RGB

    """
    max_i = 2**get_image_depth(grayscale) - 1
    rgb = np.zeros([3] + list(grayscale.shape), dtype=np.uint8)
    rgb[..., :] = grayscale / max_i * 255
    rgb = rgb.transpose(1, 2, 0)
    return rgb


def color_grayscale_image(image, rgb):
    """Color a grayscale image into an RGB image.

    Parameters:
    -----------
    image: np.array
        The grayscale image to colorify
    rgb: 3-tuple
        A tuple of RGB values from 0 to 255 representing the color that the signal in the image is to take.

    Returns:
    --------
    rgb_image: the color, 8-bit RGB image.

    """
    light_factor = image.flatten() / 2**get_image_depth(image)
    hls = colorsys.rgb_to_hls(*[c / 255 for c in rgb])

    # Write the image as HLS
    flat_image = np.zeros([np.prod(image.shape), 3])
    flat_image[:, 0] = hls[0]
    flat_image[:, 1] = light_factor
    flat_image[:, 2] = hls[2]

    # Transform back to RGB
    return (np.apply_along_axis(lambda x: colorsys.hls_to_rgb(*x), 1, flat_image) * 255).astype(np.uint8).reshape(list(image.shape) + [3])


def blend_rgb_images(*images):
    """Blend RGB images together.

    Parameters:
    ----------
    images: np.ndarray
        All the RGB images to blend.

    """
    image = np.sum(np.array(images).astype(int), axis=0)
    return image.clip(max=255).astype(np.uint8)
