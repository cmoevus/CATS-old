from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from skimage.feature import blob_log
from scipy import ndimage
from scipy.optimize import curve_fit
from math import floor, ceil
__all__ = ['find_blobs', 'fit_gaussian_on_blobs']
__detection__ = 'find_blobs'


def find_blobs(*args):
    """
    Find blobs in an image. Return a list of spots as (y, x, s, i).

    List of spots:
    y and x are the locations of the center, s is the standard deviation of the gaussian kernel and i is the intensity at the center.

    Arguments:  (as a list, for multiprocessing)
        image: a numpy array representing the image to analyze
        blur: see ndimage.gaussian_filter()'s 'sigma' argument
        threshold: see scipy.feature.blob_log()'s 'threshold' argument
        extra: information to be added at the end of the blob's properties
    """
    args = args[0] if len(args) == 1 else args
    image, extra, blur, threshold, = args[0], args[1:-1], args[-1]['blur'], args[-1]['threshold']
    b = blob_log(ndimage.gaussian_filter(image, blur), threshold=threshold, min_sigma=1)
    blobs = list()
    for y, x, s in b:
        blob = [x, y, s, image[y][x]]
        blob.extend(extra)
        blobs.append(tuple(blob))
    return fit_gaussian_on_blobs(image, blobs, args[-1]['keep_unfit'])


def fit_gaussian_on_blobs(img, blobs, keep_unfit):
    """
    Fit a Gaussian curve on the blobs in an image. Return the given list of blobs with the fitted values.

    Arguments:
        img: the ndarray of the image containing the blobs
        blobs: a list of blobs (x, y, sigma, intensity, ...) that need to go subpixel resolution. All extra information after 'intensity'  will be kept in the output.
    """
    spr_blobs = list()
    for blob in blobs:
        r = blob[2] + 1 * np.sqrt(2)
        ylim, xlim = img.shape
        y = (max(0, int(floor(blob[1] - r))), min(int(ceil(blob[1] + r)) + 1, ylim))
        x = (max(0, int(floor(blob[0] - r))), min(int(ceil(blob[0] + r)) + 1, xlim))
        data = img[y[0]:y[1], x[0]:x[1]]
        coords = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
        try:
            fit, cov = curve_fit(gaussian_2d, coords, data.ravel(), p0=(blob[3], blob[0] - x[0], blob[1] - y[0], blob[2]))
            spr_blob = [x[0] + fit[1], y[0] + fit[2], fit[3], fit[0]]
            if len(blob) > 4:
                spr_blob.extend(blob[4:])
            if x[0] <= spr_blob[0] <= x[1] and y[0] <= spr_blob[1] <= y[1]:
                spr_blobs.append(tuple(spr_blob))
            else:
                raise ValueError
        except:
            if keep_unfit == True:
                spr_blobs.append(blob)

    return spr_blobs


def gaussian_2d(coords, A, x0, y0, s):
    """Draw a 2D gaussian with given properties."""
    x, y = coords
    return (A * np.exp(((x - x0)**2 + (y - y0)**2) / (-2 * s**2))).ravel()
