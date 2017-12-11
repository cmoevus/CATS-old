# -*- coding: utf8 -*-
"""Classes and functions for deal with DNA molecules and their content."""
from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from glob import glob
import skimage
import pickle
import xml.etree.ElementTree as ET
import trackpy as tp
import itertools

import cats


class DNA(object):
    """Useful for dna molecules.

    Parameters:
    -----------
    dataset: cat.Images object
        the dataset to extract the dna from
    point: 2-tuple of 2-tuples of ints
        the positions in (x, y) of the start and end points, respectively, of the dna molecule in the dataset.
    name: string
        name of the content in the channel (optional)
    One can list as many (dataset, pt1, pt2, name) elements as they have channels, by putting them between parantheses, with an optional name to the channel

    """

    def __init__(self, *channels):
        """Set up the dna molecule."""
        if type(channels[0]) is cats.Images:  # Single channel
            channels = [channels]
        self.datasets, self.points, self.names = [], [], []
        for i, c in enumerate(channels):
            self.datasets.append(c[0])
            self.points.append(((int(round(c[1][0][0], 0)), int(round(c[1][0][1], 0))),
                               (int(round(c[1][1][0], 0)), int(round(c[1][1][1], 0)))))
            if len(c) > 2:
                self.names.append(c[2])
            else:
                self.names.append(None)

    @property
    def kymogram(self):
        """Extract the kymogram of the dna molecule from the dataset."""
        if not hasattr(self, '_kymo'):
            arguments = list()
            for c in range(len(self.datasets)):
                arguments.append((self.datasets[c], self.points[c], self.names[c]))
            self._kymo = cats.Kymogram(*arguments)
        return self._kymo

    @kymogram.setter
    def kymogram(self, k):
        self._kymo = k

    @property
    def roi(self):
        """Extract the region of interest around the location of DNA molecule."""
        #
        # Note: this sucks. I should rather use skimage.draw.polygon to draw the ROI. However, the issue is that cats.Images doesn't support advanced indexing. I should first implement advanced indexing.
        #

        # Generate the Region Of Interest from the dna
        if not hasattr(self, '_roi'):
            # Define the section size
            if not hasattr(self, 'roi_half_height'):
                self.roi_half_height = 3

            # Build the ROI object
            self._roi = list()
            for i, dataset in enumerate(self.datasets):
                x0, y0 = self.points[i][0]
                x1, y1 = self.points[i][1]
                y_max, x_max = dataset.shape
                # Consider the orientation of the data
                x_direction = -1 if x0 > x1 else 1
                if y0 > y1:
                    y_slice = slice(min(y_max, y0 + self.roi_half_height), max(0, y1 - self.roi_half_height), -1)
                else:
                    y_slice = slice(max(0, y0 - self.roi_half_height), min(y_max, y1 + self.roi_half_height), 1)
                x_slice = slice(max(0, x0), min(x1, x_max), x_direction)
                self._roi.append(dataset[:, y_slice, x_slice])
        return self._roi

    @roi.setter
    def roi(self, roi):
        self._roi = roi

    def track(self, channels=False, **kwargs):
        """Track particles from the DNA's region of interest.

        This is a convenience function wrapping the functions:
        1. locate_features
        2. filter_features
        3. link_features
        4. filter_particles

        Look into the documentation of these four functions for more details.

        This tracking method is based on Trackpy. First, it localizes blobs using
        the trackpy.batch method. Second, it filters the unwanted blobs.
        For example, those that are too close to the barriers and pedestals,
        those that are not within the vicinity of the DNA, etc. See *Parameters*
        for more details. It then links the remaining blobs into trajectories and
        proceeds to another layer of filtering to remove the spurious trajectories.

        Parameters:
        -----------
        channels: int, list of ints
            the channel(s) to track
        any keyword argument to be passed to the ``locate_features``,
            ``filter_features``, ``link_features`` and ``filter_particles``.

        """
        # Setup
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]
        if not hasattr(self, 'particles'):
            self.particles = list(None for d in self.datasets)

        # Sort the keyword arguments
        locate_kwargs = {(k, v) for k, v in kwargs.items()
                         if k in tp.batch.__code__.co_varnames}
        filter_feat_kwargs = {(k, v) for k, v in kwargs.items()
                              if k in self.filter_features.__code__.co_varnames and k != 'channels'}
        filter_parts_kwargs = {(k, v) for k, v in kwargs.items()
                               if k in self.filter_particles.__code__.co_varnames and k != 'channels'}
        link_kwargs = {(k, v) for k, v in kwargs.items()
                       if k in tp.batch.__code__.co_varnames}

        # Track
        self.locate_features(channels, **locate_kwargs)
        self.filter_features(channels, **filter_feat_kwargs)
        self.link_features(channels, **link_kwargs)
        self.filter_particles(channels, **filter_parts_kwargs)

        return [self.particles[channel] for channel in channels]

    def locate_features(self, channels=False, **kwargs):
        """Locate blobs (features, detections) in the DNA's region of interest.

        This is a wrapper for the ``trackpy.batch`` function.
        See its documentation for parameters and more information.

        Parameters:
        ----------
        channels: int, list of ints
            the channel(s) to use
        any keyword argument to be passed to the trackpy ``batch`` function

        Returns:
        --------
        particles: the list of particles for each existing channel.

        Notes:
        -----
        This modifies the `self.particles` dataframe in place.

        """
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]
        if not hasattr(self, 'particles'):
            self.particles = list(None for d in self.datasets)

        # Figure out arguments
        defaults = {
            'diameter': 3,
        }
        for k, v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        for channel in channels:
            self.particles[channel] = tp.batch(self.roi[channel], **kwargs)

        return [self.particles[channel] for channel in channels]

    def filter_features(self, channels=False, maxmass=None, max_dist_from_center=None, min_dist_from_edges=2):
        """Filter out the spurious features from the self.particles dataframe.

        Parameters:
        ----------
        channels: int, list of ints
            the channel(s) to use
        maxmass: None or number
            the maximum mass of a detection. None for no maximum.
        max_dist_from_center: None or number
            the maximum distance of the center of the particle from the vertical center of the ROI.
        min_dist_from_edges: None or number
            the minimum distance of the center of the particle from the horizontal edges of the ROI.

        Returns:
        --------
        particles: the list of particles for each existing channel.

        Notes:
        -----
        This modifies the `self.particles` dataframe in place.

        """
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]

        for channel in channels:
            d = self.particles[channel]
            if maxmass is not None:
                d = d[d['mass'] <= maxmass]
            if max_dist_from_center is not None:
                h = self.roi[channel].shape[0]
                hh = int(np.floor(h / 2))
                d = d[np.logical_and(d['y'] >= hh - max_dist_from_center,
                                     d['y'] <= hh + max_dist_from_center)]
            if min_dist_from_edges is not None:
                w = self.roi[channel].shape[1]
                d = d[np.logical_and(d['x'] >= min_dist_from_edges,
                                     d['y'] <= w - min_dist_from_edges)]
            self.particles[channel] = d

        return [self.particles[channel] for channel in channels]

    def link_features(self, channels=False, **kwargs):
        """Assemble particles (trajectories) from the features located in the DNA's region of interest.

        This is a wrapper for the ``trackpy.link_df`` function.
        See its documentation for parameters and more information.

        Parameters:
        ----------
        channels: int, list of ints
            the channel(s) to use
        any keyword argument to be passed to the trackpy ``link_df`` function. defaults: memory =5, search_range = 3

        Returns:
        --------
        particles: the list of particles for each existing channel.

        Notes:
        -----
        This modifies the `self.particles` dataframe in place.

        """
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]

        # Figure out arguments
        defaults = {
            'search_range': 3,
            'memory': 5,
        }
        for k, v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        for channel in channels:
            self.particles[channel] = tp.link_df(self.particles[channel], **kwargs)

        return [self.particles[channel] for channel in channels]

    def filter_particles(self, channels=False, min_frames=3, min_frame_ratio=1 / 3):
        """Filter out the spurious particles from the self.particles dataframe.

        Parameters:
        ----------
        channels: int, list of ints
            the channel(s) to use
        min_frames: None or int
            the minimum number of frames to be considered a particle.
        min_frame_ratio: None or float in range [0, 1]
            the minimum number of features per frame. For example, if 1/3, there must be at least 10 features detected in a trajectory that goes over 30 frames.

        Returns:
        --------
        particles: the list of particles for each existing channel.

        Notes:
        -----
        This modifies the `self.particles` dataframe in place.

        """
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]

        drop = set()
        for channel in channels:
            for i, particle in self.particles[channel].groupby('particle'):
                if min_frames is not None and len(particle) < min_frames:
                    drop.add(i)
                if min_frame_ratio is not None:
                    length = particle['frame'].max() + 1 - particle['frame'].min()
                    if len(particle) / length < min_frame_ratio:
                        drop.add(i)
            particles = self.particles[channel]
            for d in drop:
                particles = particles[particles['particle'] != d]
            self.particles[channel] = particles
        return [self.particles[channel] for channel in channels]

    def draw_particles(self, channels):
        """Draw each detection of each particle onto the kymogram."""
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]

        kymos = list()
        for channel in channels:
            kymo = self.kymogram.draw(channels=channel)
            for i, particle in self.particles[channel].groupby('particle'):
                color = cats.colors.random()
                for i, feature in particle.iterrows():
                    kymo[int(round(feature['x'], 0)), int(feature['frame'])] = color
            kymos.append(kymo)

        return kymos[0] if len(kymos) < 2 else cats.images.stack_images(*kymos)


class OldDNAStuff:

    def track(self, blur, threshold, h=3, **kwargs):
        """Track the content of the dna in 2D.

        Parameters:
        -----------
        blur: float
            See cats.detect.particles.stationary
        threshold: float
            See cats.detect.particles.stationary
        h: int
            The number of pixel to select atop and below the center of the line that defines the dna, for tracking in 2D

        """
        roi = self.to_roi(h)

        # B. Track
        particles = cats.detect.particles.stationary(roi, blur, threshold, **kwargs)
        self.particles = particles
        return particles

    def filter_particles(self):
        """Cleanup CATS dirty tracking output."""
        particles = cats.content.Particles()
        for p in self.particles:
            # Particle is more than 3px away from the edges and has more than 3 frames
            if 3 < p['x'].mean() < p.source.dimensions[0] - 3 and len(p) > 3 and len(p) / (p['t'].max() - p['t'].min()) > 0.5:
                particles.append(p)
        self.particles = particles
        return particles

    def draw_segments(self):
        """Draw each linear segment of each particle onto the dna."""
        dna = cats.extensions.draw.grayscale_to_rgb(self.pixels)
        segments = [s for p in self.segments for s in p]
        for s in segments:
            x0, x1, func, fitted = s
            x0, x1 = min(dna.shape[1] - 1, int(round(x0, 0))), min(dna.shape[1] - 1, int(round(x1, 0)))
            y0, y1 = min(dna.shape[0] - 1, int(round(func(x0), 0))), min(dna.shape[0] - 1, int(round(func(x1), 0)))
            ys, xs, line = skimage.draw.line_aa(y0, x0, y1, x1)
            dna[ys, xs, 0] = line * 255  # Red line
        return dna

    def draw_particles(self):
        """Draw each detection of each particle onto the dna."""
        # A. Rescale and transform to RGB
        dna = cats.extensions.draw.grayscale_to_rgb(skimage.exposure.rescale_intensity(self.pixels, in_range=self.intensity_scale_from_particles()))

        # B. Draw
        # detections = [(d['t'], d['x']) for p in self.particles for d in p]
        for particle in self.particles:
            color = cats.utils.colors.random()
            for detection in particle:
                dna[int(round(detection['x'], 0)), detection['frame']] = color  # Red line

        return dna

    def save(self, f=None):
        """Save the dna and its attributes into file f for future usage."""
        if f is None:
            f = self.file
        else:
            self.file = f
        with open(f, 'wb') as f:
            pickle.dump(self, f)

    def load(f):
        """Load a dna object from file f."""
        with open(f, 'rb') as f:
            return pickle.load(f)


def populate_kymograms(dnas):
    """Populate the kymograms for a list of DNA molecules that share common datasets.

    Save time on opening images by populating kymograms for several DNA molecules every time an image is opened.
    """
    # Build empty kymograms
    for dna in dnas:
        dna.kymogram.build_empty()

    # Populate kymograms per dataset
    per_datasets = itertools.groupby([(dna, channel, dataset) for dna in dnas for channel, dataset in enumerate(dna.datasets)], key=lambda x: x[2])
    for dataset, dataset_dnas in per_datasets:
        dataset_dnas = list(dataset_dnas)
        for f, frame in enumerate(dataset):
            for dna, channel, d in dataset_dnas:
                if f < dna.kymogram.shape[2]:  # Do not try to exceed the kymogram's length
                    dna.kymogram.raw[channel, :, f] = frame[dna.kymogram.pixel_positions[channel]]

    # Copy kymogram to main kymogram
    for dna in dnas:
        dna.kymogram.reset()

def xml_to_coordinates(xml_file):
    """Transform a XML ROI list from Icy into a list of dna coordinates."""
    rois = list()
    for roi in ET.parse(xml_file).getroot().findall('roi'):
        pt1 = int(round(float(roi.find('pt1').find('pos_x').text), 0)), int(round(float(roi.find('pt1').find('pos_y').text), 0))
        pt2 = int(round(float(roi.find('pt2').find('pos_x').text), 0)), int(round(float(roi.find('pt2').find('pos_y').text), 0))
        rois.append((pt1, pt2))
    return rois


def csv_to_coordinates(csv_file):
    """Transform a CSV lines list from ImageJ into a list of dna coordinates.

    How to use:
    -----------
    - Draw a line over the DNA
    - Measure it (Ctrl+M)
    - Make sure you have the following measurements set:
        - Boundary rectangle
    - Save with the column headers on.
    """
    lines = pd.read_csv(csv_file)
    lines['EX'] = lines['BX'] + lines['Width']
    lines['EY'] = lines['BY'] - lines['Height'] * lines['Angle'] / abs(lines['Angle'])
    return [((r[1]['BX'], r[1]['BY']), (r[1]['EX'], r[1]['EY'])) for r in lines.iterrows()]


def generate_registration_function(reference, relative):
    """Generate a registration function that transforms coordinates (x, y) from a reference source into (x, y) to the relative source.

    Parameters:
    -----------
    reference: 2-tuple
        the coordinates (x0, y0) of the reference point
    relative: 2-tuple
        the coordinates (x1, y1) of the relative point

    Returns:
    --------
    register: func
        a function that transforms reference points into relative points

    """
    dx = relative[0] - reference[0]
    dy = relative[1] - reference[1]

    def register(x, y):
        """Transform coordinates (x,y) of the reference into (x,y) of the relative dataset."""
        return x + dx, y + dy

    return register


def invert_coordinates(coords):
    """Invert the orientation of a list of kymogram coordinates, so that the top of the kymogram becomes the bottom."""
    return [(pt2, pt1) for pt1, pt2 in coords]


def register_coordinates(reg_func, coords):
    """Register a list of coordinates as pt1, pt2."""
    coords = np.array(coords.copy())
    coords[:, 0, 0], coords[:, 0, 1] = reg_func(coords[:, 0, 0], coords[:, 0, 1])
    coords[:, 1, 0], coords[:, 1, 1] = reg_func(coords[:, 1, 0], coords[:, 1, 1])
    return coords


def hand_picked_files_to_df(path):
    """Transform your loosy .csv files from ImageJ into a magnificient pandas DataFrame."""
    particles = list()
    for i, f in enumerate(glob(path)):
        values = pd.read_csv(f, header=None)
        values.rename(columns={0: 'x', 1: 'y'}, inplace=True)
        values['id'] = i
        particles.append(values)
    return pd.concat(particles) if len(particles) > 0 else pd.DataFrame([], columns=['x', 'y', 'id'])
