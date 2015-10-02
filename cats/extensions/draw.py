from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage import io, draw, exposure
from colorsys import hls_to_rgb
from random import randrange



#
# Functions related to Particles
#

def particles_perimeters_by_frame(particles, nb_frames, shape):
    """
    Transform a list of particles into a list, organized by frames, of colored circles that surround each particle.

    Arguments:
        particles: the list of particles
        nb_frames: the total number of frames in the source.
        shape: the shape of the source.
    """
    frames = [[] for f in range(nb_frames)]
    colors = [get_color() for i in particles]
    for particle, color in zip(particles, colors):
        sigma = int(np.mean([s['s'] for s in particle]))
        for s in particle:
            area = draw.circle_perimeter(int(s['y']), int(s['x']), sigma, shape=shape)
            frames[s['t']].append((area, color))
    return frames


def intensity_scale_from_particles(particles):
    """Make an intensity scale for skimage.exposure.rescale_intensity based on particles intensity."""
    i = [s['i'] for p in particles for s in p]
    if len(i) > 1:
        m, s = np.mean(i), np.std(i)
        scale = (m - 3 * s, m + 3 * s)
    else:
        raise ValueError('Not enough particles to get intensity scale.')
    return scale


def choose_scale_for_particles(scale, particles, source):
    """Finds the ideal scale between user's will and actual limitations."""
    if scale == True:
        try:
            scale = intensity_scale_from_particles(particles)
        except:
            scale = intensity_scale_from_images(source)
    try:
        l = len(scale)
        if l != 2:
            raise TypeError
    except TypeError:
        scale = intensity_scale_from_images(source)
    return scale


def draw_particles(self, output=None, source=0, rescale=True):
    """
    Draw the particles from the given source on its images.

    Argument:
        output: the directory in which to write the files. If None, returns the images as a list of arrays.
        source: the index of the source, in the source list, to draw.
        rescale: adjust intensity levels to the given tuple (lower bound, upper bound) If True, will adjust the intensity levels to that of the detected particles. Anything else will adjust intensity to the images.
    """
    source = self.sources[source]
    particles = [i for i in self if i.source == source]
    perimeters = particles_perimeters_by_frame(particles, source.length, source.shape)
    scale = choose_scale_for_particles(rescale, particles, source)
    draw_on_source(source, perimeters, output, scale)


def draw_particle(self, output=None, source=0, rescale=True):
    """
    Draw the particle on its source images.

    Argument:
        output: the directory in which to write the files. If None, returns the images as a list of arrays.
        source: the index of the source, in the source list, to draw.
        rescale: adjust intensity levels to the given tuple (lower bound, upper bound) If True, will adjust the intensity levels to that of the detected particles. Anything else will adjust intensity to the images.
    """
    source = self.source
    particles = [self]
    perimeters = particles_perimeters_by_frame(particles, source.length, source.shape)
    scale = choose_scale_for_particles(rescale, particles, source)
    draw_on_source(source, perimeters, output, scale)

#
# General functions
#

def intensity_scale_from_images(source):
    """Make an intensity scale for skimage.exposure.rescale_intensity based on images."""
    return source.min_intensity(), source.max_intensity()


def grayscale_to_rgb(grayscale, max_i=65535):
    """
    Transform a greyscale image to a 8bits RGB.

    Arguments:
        grayscale: the image to transform into 8bit RGB
        max_i: the maximum value of the grayscale to consider as the maximum value of the RGB image. Default is 65535 (2**16 - 1), the absolute maximal value for a 16bit grayscale image.
    """
    rgb = np.zeros([3] + list(grayscale.shape), dtype=np.uint8)
    rgb[..., :] = grayscale / max_i * 255
    rgb = rgb.transpose(1, 2, 0)
    return rgb


def get_color():
    """Return a random color."""
    return tuple(int(round(i * 255, 0)) for i in hls_to_rgb(randrange(0, 360) / 360, randrange(50, 90, 1) / 100, randrange(30, 80, 10) / 100))


def draw_on_source(source, areas, output=None, scale=(0, 65535)):
    """
    Draw content from source in output.

    Arguments:
        source: the source images as a cats.sources.{Images, ROI} object.
        areas: the list of regions to draw (as a 2-tuple of region, color in 3-tuple RGB), organized by frames
        output: the directory to write into.  If None, the drawn images are returned as a list.
        scale: the intensity levels to adjust images to.
    """
    for t, image in enumerate(source.read()):
        if output is None:
            images = list()

        # A. Rescale
        image = exposure.rescale_intensity(image, in_range=scale)

        # B. Transform to RGB
        rgb = grayscale_to_rgb(image)

        # C. Draw areas on source
        for p, color in areas[t]:
            rgb[p[0], p[1], :] = color

        # Show/save image
        if output is None:
            images.append(rgb)
        else:
            io.imsave("{0}/{1}.tif".format(output, t), rgb)

    return images if output is None else None

__extension__ = {
    'Particles': {'draw': draw_particles},
    'Particle': {'draw': draw_particle},
}
