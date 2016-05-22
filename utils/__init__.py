"""Useful functions for various CATS needs."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from math import ceil
import skimage


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


class rdict(dict):
    """
    Recursive dict.

    Recursively looks for keys and returns None if key does not exist.
    """

    def __init__(self, d, parent=None):
        """Recursively transforms dict in into rdicts."""
        dict.__init__(self, d)
        for key, item in self.items():
            if type(item) == dict:
                self[key] = rdict(item, self)
        self.__parent__ = parent

    def __getitem__(self, item):
        """Look recursively for items."""
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            if self.__parent__ is not None:
                return self.__parent__[item]
            else:
                return None


def find_barriers(images, dbp=47, dbb=139, approx=20, orientation='pb', axis='y'):
    """
    Find the barriers in an images.

    Arguments:
        images: cats.sources.ROI or cats.sources.Images object
        dbp: Distance between the barriers and the pedestals in pixels
        dbb: Distance between two sets of barriers, in pixels
        axis: The axis of the image with which the barriers are parallel. Either 'x' or 'y'
        approx: max offset distance between the expected and detected barriers.
    """
    # Make a projection of the image
    img = np.zeros(images.shape)
    for i in images.read():
        img += i
    img /= images.length
    img = sp.ndimage.filters.gaussian_filter(img, 2)
    axis = 0 if axis == 'y' else 1
    projection = np.sum(img, axis=axis)

    # Find peaks
    d_proj = np.diff(projection)
    tangents = list()
    for i in range(0, len(d_proj) - 1):
        neighbors = sorted((d_proj[i], d_proj[i + 1]))
        if neighbors[0] < 0 and neighbors[1] > 0:
            tangents.extend([i])
    extremas = np.where((np.diff(d_proj)) - (d_proj[:-1]) >= 0)[0]
    peaks = sorted([p for p in tangents if p in extremas])
    distances = np.subtract.outer(peaks, peaks)

    # Build sets of barriers
    max_sets = int(ceil(len(projection) - dbp) / dbb) + 1
    exp_dists, sets = [dbp], list()
    for i in range(1, max_sets):
        exp_dists.extend([i * dbb - dbp, i * dbb])

    # Find the best possible match for consecutive barriers-pedestals
    # for each peak
    for peak in range(distances.shape[0]):
        i = 0  # Position in the sets of barriers
        j, last_j = 0, -1  # Distance with last peak
        barriers_set, current_set = list(), list()
        a = 0  # Allowed difference for ideal distance
        while i < max_sets and j < len(exp_dists):

            # Update the distances to search at
            if j != last_j:
                search = [exp_dists[j]]
                for k in range(1, approx):
                    search.extend([exp_dists[j] - k, exp_dists[j] + k])
                last_j = j

            # Try to find a pair
            match = np.where(distances[peak] == search[a])[0]
            if len(match) != 0:
                if j == 0:
                    current_set = (peaks[peak], peaks[match[0]])
                    barriers_set.append((current_set, a))
                else:
                    i += j // 2 + 1
                    if len(current_set) == 1:
                        barriers_set.append((current_set, approx + 1))
                    current_set = [match[0]]
                peak = match[0]
                j = 0 if j % 2 == 1 else 1
                a = 0

            # No pair found: look for it with more laxity
            else:
                a += 1

            # This pair does not exists in all permitted limits.
            # Look for next pair.
            if a == approx:
                a = 0
                j += 1

        if len(barriers_set) > 0:
            sets.append(barriers_set)

    set_scores = list()
    for s in sets:
        score = sum([n[1] for n in s])
        set_scores.append((len(s), -score))

    barriers = sorted([sorted(n[0]) for n in sets[set_scores.index(sorted(set_scores, reverse=True)[0])]])

    if orientation == 'pb':
        correct_barriers = list()
        for b in barriers[::-1]:
            correct_barriers.append((b[1], b[0], -1))
        return correct_barriers
    else:
        return barriers


def draw_barriers(images, barriers, axis='y'):
    """Draw the result of find_barriers on images."""
    plt.figure()
    plt.axis('off')
    # Image projection
    img = np.zeros(images.shape)
    for i in images.read():
        img += i
    img /= images.length
    img = skimage.exposure.rescale_intensity(img, in_range=(img.mean() - 3 * img.std(), img.mean() + 3 * img.std()))
    plt.imshow(img, cmap=plt.cm.gray)
    for b in barriers:
        b1, b2 = b[0], b[1]
        f = plt.axvline if axis == 'y' else plt.axhline
        f(b1, color='red')
        f(b2, color='red')
    plt.show()
