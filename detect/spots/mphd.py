# -*- coding: utf8 -*-
"""
Detect domes in the given images.

Implementation of the algorithm as presented in:
S. H. Rezatofighi, R. Hartley, and W. E. Hughes, “A new approach for spot detection in total internal reflection fluorescence microscopy,” in 2012 9th IEEE International Symposium on Biomedical Imaging (ISBI), 2012, pp. 860–863.

MPHD stands for 'Maximum Possible Height Dome'.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import ndimage
import skimage as sk
from ...utils import get_image_depth


def mphd(images, Dr=3, h=1, smoothing=0.5):
    """
    Detect domes in the given images.

    THIS DOES NOT WORK, PROBABLY FOR TWO REASONS:
        - THE NOISE IS NOT PROPERLY REMOVED (RADIUS OF A SPOTS NOT << RADIUS NOISE)
        - OBP FINDING IS NOT PROPERLY WORKING. EACH REGION SHOULD BE DELIMITED SO THAT IT ONLY CONTAINS ONE MAXIMA AND ENCOMPASS ALL PIXELS THAT ARE BETWEEN THE SURROUNDING MINIMA.

    images: source Images as a cats.Images object
    Dr: int bigger than the radius of a detection but smaller than the radius of a noise region.
    h = the height...
    smoothing: sigma of the gaussian for smoothing the image. <= sigma of a detection.
    """
    for image in images.read():
        # Smooth the image
        # S = ndimage.gaussian_filter(image, smoothing)
        S = image

        # Find regional maxima
        dh = S.astype(np.int32) - h
        marker = np.where(dh < 0, 0, dh)
        maxima = (S - sk.morphology.reconstruction(marker, S)).astype(bool)

        # # Find regional minima
        # dh = S.astype(np.int32) + h
        # marker = np.where(dh == 2**get_image_depth(image), 2**get_image_depth(image) - 1, dh)
        # minima = (sk.morphology.reconstruction(marker, S, 'erosion') - S).astype(bool)

        # Prepare the adaptive mask
        Ma = marker.copy()
        # Ma = np.zeros(image.shape, dtype=image.dtype)
        for y, x in zip(*np.where(maxima)):
            sx = slice(max(0, x - Dr), min(x + Dr + 1, image.shape[1]))
            sy = slice(max(0, y - Dr), min(y + Dr + 1, image.shape[0]))
            region = S[sy, sx].astype(np.int32)
            r_x, r_y = x - sx.start, y - sy.start
            p0 = find_obp(region, r_x, r_y, Dr)
            rMa = region - region[r_y][r_x] + p0
            rMA = np.where(rMa < 0, 0, rMa)
            Ma[sy, sx] = np.where(Ma[sy, sx] < rMa, Ma[sy, sx], rMa)
            # Ma[sy, sx] = rMa

        # Find maxima using the adaptive mask
        spots = (S - sk.morphology.reconstruction(Ma - 1, S)).astype(bool)
        # return S, sk.morphology.remove_small_objects(spots, 2)
        return S, spots


def find_obp(region, x, y, Dr):
    """
    Find the optimal base pixel in the given regional base.

    region: the regional base
    x, y: coordinates of the pixel to start from
    Dr: max radius to look at
    """
    maxima = list()
    for r in range(1, Dr + 1):
        boundary = sk.draw.circle_perimeter(y, x, r, 'andres', shape=region.shape)
        maxima.append(max(region[boundary]))
    return min(maxima)
