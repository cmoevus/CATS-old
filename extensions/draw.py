"""
Draw content onto source images.

Module implements drawing of
    - Particles (list or single)

The module adds the draw() method to the following objects:
    - Particles
    - Particle

Arguments of the draw() method:
    output: the directory in which to write the files. If None, returns the images as a list of arrays.
    source: the index of the source, in the source list, to draw.
    rescale: adjust intensity levels to the given tuple (lower bound, upper bound) If True, will adjust the intensity levels to that of the detected particles. Anything else will adjust intensity to the images.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from skimage import io, draw, exposure
from ..utils import colors, get_image_depth


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
    colours = [colors.random() for i in particles]
    for particle, color in zip(particles, colours):
        r_x = int(particle['sx'].mean() * np.sqrt(2)) + 2
        r_y = int(particle['sy'].mean() * np.sqrt(2)) + 2
        # r = max(r_x, r_y)
        for s in particle:
            area = draw.ellipse_perimeter(int(s['y']), int(s['x']), r_y, r_x, shape=shape)
            # area = draw.circle_perimeter(int(s['y']), int(s['x']), r, shape=shape)
            frames[s['t']].append((area, color))
    return frames


def intensity_scale_from_particles(particles):
    """Make an intensity scale for skimage.exposure.rescale_intensity based on particles intensity."""
    i = [s['i'] for p in particles for s in p]
    if len(i) > 1:
        m, s = np.median(i), np.std(i)
        scale = (m - 3 * s, m + 3 * s)
    else:
        raise ValueError('Not enough particles to get intensity scale.')
    return scale


def choose_scale_for_particles(scale, particles, source):
    """Find the ideal scale between user's will and actual limitations."""
    if scale == True:
        scale = intensity_scale_from_images(source)
        try:
            scale_p = intensity_scale_from_particles(particles)
            scale = (max(scale[0], scale_p[0]), min(scale[1], scale_p[1]))
        except:
            pass
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


def draw_particle(self, output=None, rescale=True):
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
# Functions related to filaments
#
def filaments_by_frames(filaments, nb_frames, shape):
    """
    Transform a list of filaments into a list, organized by frames, of lines that hover each particle.

    Arguments:
        particles: the list of particles
        nb_frames: the total number of frames in the source.
        shape: the shape of the source.
    """
    frames = [[] for f in range(nb_frames)]
    colours = [colors.random() for i in filaments]
    for filament, color in zip(filaments, colours):
        for f in filament:
            x1, y1, a, l = int(f['x']), int(f['y']), f['a'], f['l'] - 1
            x2, y2 = int(np.cos(a) * l + x1), int(np.sin(a) * l + y1)
            x2, y2 = min(shape[1] - 1, x2), min(shape[0] - 1, y2)
            area = draw.line(y1, x1, y2, x2)
            frames[f['t']].append((area, color))
    return frames


def draw_filaments(self, output=None, source=0, scale=None):
    """
    Draw the filaments from the given source on its images.

    Argument:
        output: the directory in which to write the files. If None, returns the images as a list of arrays.
        source: the index of the source, in the source list, to draw.
        scale: adjust intensity levels to the given tuple (lower bound, upper bound). Anything else will adjust intensity to the images.
    """
    source = self.sources[source]
    filaments = self
    lines = filaments_by_frames(filaments, source.length, source.shape)
    scale = intensity_scale_from_images(source) if scale is None else scale
    draw_on_source(source, lines, output, scale)


def draw_filament(self, output=None, scale=None):
    """
    Draw the filaments from the given source on its images.

    Argument:
        output: the directory in which to write the files. If None, returns the images as a list of arrays.
        source: the index of the source, in the source list, to draw.
        scale: adjust intensity levels to the given tuple (lower bound, upper bound). Anything else will adjust intensity to the images.
    """
    source = self.source
    filaments = [self]
    lines = filaments_by_frames(filaments, source.length, source.shape)
    scale = intensity_scale_from_images(source) if scale is None else scale
    draw_on_source(source, lines, output, scale)


#
# General functions
#


def intensity_scale_from_images(source):
    """Make an intensity scale for skimage.exposure.rescale_intensity based on images."""
    return source.min_intensity(), source.max_intensity()


def grayscale_to_rgb(grayscale):
    """
    Transform a greyscale image to a 8bits RGB.

    Arguments:
        grayscale: the image to transform into 8bit RGB
        depth: the depth of the grayscale image. Default is 65535 (2**16 - 1), the absolute maximal value for a 16bit grayscale image.
    """
    max_i = 2**get_image_depth(grayscale) - 1
    rgb = np.zeros([3] + list(grayscale.shape), dtype=np.uint8)
    rgb[..., :] = grayscale / max_i * 255
    rgb = rgb.transpose(1, 2, 0)
    return rgb


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
    'Filaments': {'draw': draw_filaments},
    'Filament': {'draw': draw_filament},
}
