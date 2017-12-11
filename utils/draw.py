"""Utilities for drawing stuff."""
import skimage
import skimage.draw
import numpy as np


def dashed_line(r0, c0, r1, c1, full=3, empty=3):
    """Just like skimage.draw.line, but the line will be dashed."""
    line = skimage.draw.line(r0, c0, r1, c1)
    l = len(line[0])
    keep = list()
    for i in range(full):
        keep.extend(list(range(i, l, full + empty + 1)))
    keep = sorted(keep)
    return line[0][keep], line[1][keep]


def make_dashed(x, y, full=3, empty=3):
    """Make an (x, y) pair of points dashed, assuming they are initially continuous."""
    l = len(x)
    keep = list()
    for i in range(full):
        keep.extend(list(range(i, l, full + empty + 1)))
    keep = sorted(keep)
    return x[keep], y[keep]


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
        width, height = dims[:, 1].max(), sum(dims[:, 0]) + len(dims) * spacing
    else:
        height, width = dims[:, 0].max(), sum(dims[:, 1]) + len(dims) * spacing
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
