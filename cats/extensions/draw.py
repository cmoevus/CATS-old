from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage import io, draw, exposure
from colorsys import hls_to_rgb
from random import randrange


def draw_particles(self, output=None, source=0, rescale=True):
    """
    Draw the tracks and/or spots onto the images of the dataset as RGB images.

    Argument:
        output: the directory in which to write the files. If None, returns the images as a list of arrays.
        source: the index of the source, in the source list, to draw.
        rescale: adjust intensity levels to the detected content
    """
    if output is None:
        images = list()

    source = self.sources[source]
    particles = [i for i in self if i.source == source]

    # Make colors
    colors = [np.array(hls_to_rgb(randrange(0, 360) / 360, randrange(50, 90, 1) / 100, randrange(20, 80, 10) / 100)) * 255 for i in particles]

    # Separate particles per image
    frames = [[] for f in range(source.length)]
    for particle, color in zip(particles, colors):
        sigma = int(np.mean([s['s'] for s in particle]))
        for s in particle:
            spot = (int(s['y']), int(s['x']), sigma, color)
            frames[s['t']].append(spot)

    # Calculate the intensity range
    if rescale == True:
        i = [s['i'] for p in particles for s in p]
        if len(i) != 0:
            m, s = np.mean(i), np.std(i)
            scale = (m - 3 * s, m + 3 * s)
        else:
            scale = False

    for t, image in enumerate(source.read()):
        # Prepare the output image (a 8bits RGB image)
        ni = np.zeros([3] + list(source.shape), dtype=np.uint8)

        if rescale is True and scale is not False:
            image = exposure.rescale_intensity(image, in_range=scale)

        ni[..., :] = image / image.max() * 255
        ni = ni.transpose(1, 2, 0)

        # Mark particles
        for s in frames[t]:
            area = draw.circle_perimeter(s[0], s[1], s[2], shape=source.shape)
            ni[area[0], area[1], :] = s[3]

        # Show/save image
        if output is None:
            images.append(ni)
        else:
            io.imsave("{0}/{1}.tif".format(output, t), ni)

    return images if output is None else None

__extension__ = {'Particles': {'draw': draw_particles}}
