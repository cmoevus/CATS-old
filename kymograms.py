# -*- coding: utf8 -*-
"""Draw and assemble kymograms."""
from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
import scipy as sp
from glob import glob
import skimage
import pickle
import xml.etree.ElementTree as ET
import pims
import colorsys
import numbers

import cats.utils.math
import cats.utils
import cats.colors
import cats.extensions.draw
import cats.detect
import cats.content
import cats.images


class Kymogram:
    """Useful for kymograms.

    Parameters:
    -----------
    dataset: cat.ROI object
        the dataset to extract the kymogram from
    pt1, pt2: tuples of ints
        the positions (x, y) of the start and end points of the kymogram in the dataset. Note that the lenght must be identical in all channels.
    name: string
        name of the content in the channel (optional)
    One can list as many (dataset, pt1, pt2, name) elements as they have channels, by putting them between parantheses, with an optional name to the channel.

    """

    def __init__(self, *channels):
        """Set up the kymogram."""
        if type(channels[0]) is cats.Images:  # Single channel
            channels = [channels]
        self.pixels = None  # The kymogram
        self.raw = None  # The raw, unmodified kymogram
        self.datasets, self.points, self.names, self.pixel_positions, self.pixels = [], [], [], [], None
        for i, c in enumerate(channels):
            self.datasets.append(c[0])
            self.points.append(((int(round(c[1][0][0], 0)), int(round(c[1][0][1], 0))),
                               (int(round(c[1][1][0], 0)), int(round(c[1][1][1], 0)))))
            self.pixel_positions.append(skimage.draw.line(self.points[i][0][1], self.points[i][0][0], self.points[i][1][1], self.points[i][1][0]))
            if len(c) > 2:
                self.names.append(c[2])
            else:
                self.names.append(None)

    @property
    def shape(self):
        """Return the shape of the kymogram, as (number of channels, DNA length, number of frames)."""
        if self.raw is not None:
            return self.raw.shape
        else:
            length = min([len(d) for d in self.datasets])
            return len(self.datasets), len(self.pixel_positions[0][0]), length

    def build_empty(self):
        """Build the shape of the kymogram, without any data in it."""
        frames = min([len(d) for d in self.datasets])
        self.raw = np.zeros((len(self.datasets), len(self.pixel_positions[0][0]), frames), dtype=np.uint16)

    def build(self):
        """Extract the kymograms from the datasets."""
        self.build_empty()  # Build the structure
        length = self.shape[2]
        for i, dataset in enumerate(self.datasets):
            dataset = self.datasets[i]
            for j in range(length):
                frame = dataset[j]
                self.raw[i, :, j] = frame[self.pixel_positions[i]]
        self.pixels = self.raw.copy()
        return self.pixels

    def get(self, channels=False):
        """Extract the kymogram from the dataset.

        Parameters:
        -----------
        channels: list of numbers
            the channels to use for the kymogram. If False, uses all channels.

        Returns:
        --------
        Kymogram in the shape (channel, position, frame)

        """
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]
        if self.pixels is None:
            self.build()
        return np.squeeze(self.pixels[channels])

    def reset(self):
        """Reset the scaling and modifications to the kymogram."""
        self.pixels = self.raw.copy()

    def rescale(self, scales='image', channels=False):
        """Rescale the kymogram to the desired values.

        Parameters:
        -----------
        scales: list of 2-tuples
            The min and max value for each channel. If same for all channels, one can input a 2-tuple only.

        """
        # Check the input
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]
        if isinstance(scales[0], numbers.Number) or isinstance(scales, str):
            scales = [scales for c in channels]

        # Do
        for c in channels:
            self.pixels[c] = skimage.exposure.rescale_intensity(self.pixels[c], in_range=scales[c])
        return self.pixels

    def __repr__(self):
        """Representation of the kymogram."""
        return self.pixels

    def draw(self, colors=None, channels=False):
        """Draw the kymogram as an RGB image.

        Parameters:
        -----------
        colors: list of 3-tuples
            the RGB colors to use for each channel. If None, all channels will be white. The range for the colors goes from 0 to 1, like for matplotlib, and not 0, to 255 like usual RGB.
        channels: list of numbers
            the channels to use for the kymogram. If False, uses all channels.

        """
        if self.pixels is None:
            self.get()
        if channels is False:
            channels = range(len(self.datasets))
        elif type(channels) is int:
            channels = [channels]
        if colors is None:
            colors = [(1, 1, 1) for i in channels]
        images = [cats.images.color_grayscale_image(self.pixels[c], colors[i]) for i, c in enumerate(channels)]
        return cats.images.blend_rgb_images(*images) if len(images) > 1 else images[0]


class OldKymoStuff:

    def scaled_to_content(self):
        """Return the kymogram, scaled to its content."""
        if self.pixels is None:
            kymo = self.get()
        else:
            kymo = self.pixels
        return skimage.exposure.rescale_intensity(kymo, in_range=self.intensity_scale_from_particles())

    def draw(self, colors=None, channels=False):
        """Draw the kymogram as an RGB image.

        Parameters:
        -----------
        colors: list of 3-tuples
            the RGB colors to use for each channel. If None, all channels will be white. The range for the colors goes from 0 to 1, like for matplotlib, and not 0, to 255 like usual RGB.
        channels: list of numbers
            the channels to use for the kymogram. If False, uses all channels.

        """
        if self.pixels is None:
            self.get()
        if not channels:
            channels = range(len(self.datasets))
        if colors is None:
            colors = [(1, 1, 1) for i in channels]
        max_pixel_value = 2**cats.utils.get_image_depth(self.pixels)
        shape = self.pixels.shape[1], self.pixels.shape[2], 3
        images = list()
        for c in channels:
            hls = colorsys.rgb_to_hls(*colors[c])
            source = self.pixels[c, :, :].flatten()
            max_pixel_value = max(source)
            image = np.array([colorsys.hls_to_rgb(hls[0], hls[1] * p / max_pixel_value, hls[2]) for p in source])
            images.append(image.reshape(shape))
        return cats.colors.blend_rgb_images(*images)

    def load_hand_picked_particles_from_files(self, path):
        """Take in the path to a kymogram and transforms into an array and a list of particles suitable for the by_hand() function.

        Assumes that you label the particle point lists as the name of the kymogram + " - n.csv", where n is the particle number.

        Parameters:
        ----------
        path: str
            The path (as readable by glob) to the CSV files describing the particles

        Returns:
        -------
        particles: list
            A list of lists of coordinate, as follow:
                [
                particle1 [(x1, y1), (x2, y2), etc.],
                particle2 [(x1, y1), (x2, y2), ..., (xn, yn)],
                ...,
                particlen [...]
                ]
            With the coordinates being the first and last point of each linear segment of the trace.

        """
        particles = list()
        for f in glob(path):
            values = pd.read_csv(f, header=None).values
            particles.append(values)
        self.hand_picked_particles = particles
        return particles

    def load_hand_picked_particles_from_df(self, df):
        """Load hand picked particles from a DataFrame.

        Parameters:
        -----------
        df: pandas.DataFrame
            The dataframe with columns x, y, id, where each row is a position x, y on the kymogram for particle number 'id'.

        Returns:
        --------
        particles: list
            See load_hand_picked_particles_from_files.

        """
        ps = df.groupby('id')
        particles = list()
        for i, p in ps:
            particles.append([(s['x'], s['y']) for i, s in p.iterrows()])
        self.hand_picked_particles = particles
        return particles

    def track_hand_picked_particles(self, h=3, threshold=0.95):
        """Track particles onto the kymogram from the given files.

        Fit lines within each set of points (particles) for the given kymogram.

        Parameters:
        -----------
        h: The number of pixel atop and below the center of the particle, for fitting
        threshold: Minimum R2 value for quality of fit, fits with R2 values below are rejected.

        Returns:
        --------
        fitted_particles: Particles object
            A list of Particle objects, containing:
            - t: the position in time
            - x: the position in x (fitted)
            - i: the intensity at maximum
            - sx: the standard deviation of the signal gaussian, in x
        segment_equations: list
            a list of particles with fitted segments, as follow:
                start, end, linear equation, whether the fit was successful
            if the fit was not successful, returns the slope based on the input points

        """
        # Generate the kymogram, if need be
        if self.pixels is None:
            self.get()

        segments, fitted_particles = list(), cats.content.Particles()
        for particle in self.hand_picked_particles:
            # 1. Get the data for each segment
            detections, equations = list(), list()
            for i in range(len(particle) - 1):
                # a. Determine the region of interest
                # But first, remove the subpixel information for slicing
                pts = [int(round(v, 0)) for p in (particle[i], particle[i + 1]) for v in p]
                xs = [min(v, len(self.dataset) - 1) for v in (pts[0], pts[2])]
                ys = [min(v, len(self.pixel_positions[0]) - 1) for v in (pts[1], pts[3])]
                pt1, pt2 = (xs[0], ys[0]), (xs[1], ys[1])
                m, b = cats.utils.math.line_from_points(pt1, pt2)

                # b. Locate precisely the position at each frame
                segment_detections = list()
                for x in range(pt1[0], pt2[0] + 1):
                    # i. Get the pixel values
                    y = int(m * x + b)
                    lower_bound = max(y - h, 0)
                    upper_bound = min(y + h + 2, self.pixels.shape[0] - 1)
                    values = self.pixels[lower_bound:upper_bound, x]
                    window_y = y - lower_bound

                    # ii. Fit a gaussian
                    x_fit = range(len(values))
                    try:
                        fit = sp.optimize.curve_fit(cats.utils.math.noisy_gaussian,
                                                    x_fit,
                                                    values,
                                                    p0=(values[window_y], window_y, 2, np.mean(values)),
                                                    # bounds=([np.mean(values), max(0, window_y - 2), 0, 0], [np.max(values), min(window_y + 2, len(values) - 1), np.inf, np.mean(values)])
                                                    bounds=([0, max(0, window_y - 2), 0, 0], [np.inf, min(window_y + 2, len(values) - 1), np.inf, np.inf])
                                                    )[0]
                    except (RuntimeError, TypeError):
                        fit = False

                    # iii. Determine the quality of the fit using the coefficient of determination
                    # (https://stackoverflow.com/a/29015491)
                    if type(fit) != bool:
                        y_fit = cats.utils.math.noisy_gaussian(x_fit, *fit)
                        ss_res = np.sum((values - y_fit) ** 2)
                        ss_tot = np.sum((values - np.mean(values)) ** 2)
                        r2 = 1 - (ss_res / ss_tot)
                    else:
                        r2 = 0

                    # iv. Decide whether to keep the point
                    if r2 >= threshold:
                        segment_detections.append((x, lower_bound + fit[1], values[min(int(round(fit[1], 0)), len(values) - 1)], fit[2]))  # In order: t, x, i, sx

                    # Make sure the boundaries of the segment are in the detections
                    elif r2 < threshold and x in (pt1[0], pt2[0]):
                        segment_detections.append((x, y, values[int((upper_bound - lower_bound) / 2)], 0))  # In order: t, x, i, sx

                # c. Add detections to the list
                detections.extend(segment_detections)

                # d. Fit a line through the nicely positioned points
                if len(segment_detections) < 2:
                    segment_detections = [s[:2] for s in segment_detections]
                    segment_detections.extend([pt1, pt2])
                    fitted = False
                else:
                    fitted = True
                # d.1. Get the equation
                centers = np.array(segment_detections)
                xs, ys = centers[:, 0], centers[:, 1]
                equation = np.poly1d(np.polyfit(centers[:, 0], centers[:, 1], 1))
                # d.2 Check the quality of the fit
                y_fit = equation(xs)
                ss_res = np.sum((ys - y_fit) ** 2)
                ss_tot = np.sum((ys - np.mean(ys)) ** 2)
                r2_eq = 1 - (ss_res / ss_tot)
                if r2_eq >= threshold:  # Use the fit, it's good quality
                    equations.append((pt1[0], pt2[0], equation, fitted))
                else:  # Use the user-defined points, they're probably better...
                    equations.append((pt1[0], pt2[0], np.poly1d([m, b]), False))

            # 2. Save this particle
            p = np.array(detections, dtype=[('t', int), ('x', float), ('i', int), ('sx', float)]).view(cats.content.Particle)
            p.source = self.dataset
            fitted_particles.append(p)
            segments.append(equations)

        self.particles = fitted_particles
        self.segments = segments
        return fitted_particles, segments

    def define_segments_from_hand_picked_particles(self, floor_x=True):
        """Define segments from hand picked particles.

        Parameters:
        -----------
        round_x: bool
            If True, the x values (time) will be floored to the lower integer. This helps for consistency with subsequent steps.

        """
        segments = list()
        for particle in self.hand_picked_particles:
            equations = list()
            for i in range(len(particle) - 1):
                pt1, pt2 = particle[i], particle[i + 1]
                if floor_x:
                    pt1 = int(pt1[0]), pt1[1]
                    pt2 = int(pt2[0]), pt2[1]
                equations.append((pt1[0], pt2[0], np.poly1d(cats.utils.math.line_from_points(pt1, pt2)), False))
            segments.append(equations)
        self.segments = segments
        return segments

    def to_roi(self, h=3):
        """Generate a 2D region of interest from the location of the kymogram."""
        # A. Generate the Region Of Interest from the kymogram
        x0, y0 = self.start
        x1, y1 = self.end
        y_max, x_max = self.dataset.shape
        x2 = -1 if x0 > x1 else 1  # Consider the orientation of the data
        y2 = -1 if y0 > y1 else 1  # Consider the orientation of the data
        min_y, max_y = max(0, min(y0, y1) - h), min(max(y0, y1) + h + 1, y_max)
        roi = cats.sources.ROI(self.dataset, x=(max(0, x0), min(x1, x_max), x2), y=(min_y, max_y, y2))
        roi.dna_y = min(y0, y1) - max(0, min(y0, y1) - h)
        return roi

    def track(self, blur, threshold, h=3, **kwargs):
        """Track the content of the kymogram in 2D.

        Parameters:
        -----------
        blur: float
            See cats.detect.particles.stationary
        threshold: float
            See cats.detect.particles.stationary
        h: int
            The number of pixel to select atop and below the center of the line that defines the kymogram, for tracking in 2D

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
        """Draw each linear segment of each particle onto the kymogram."""
        kymo = cats.extensions.draw.grayscale_to_rgb(self.pixels)
        segments = [s for p in self.segments for s in p]
        for s in segments:
            x0, x1, func, fitted = s
            x0, x1 = min(kymo.shape[1] - 1, int(round(x0, 0))), min(kymo.shape[1] - 1, int(round(x1, 0)))
            y0, y1 = min(kymo.shape[0] - 1, int(round(func(x0), 0))), min(kymo.shape[0] - 1, int(round(func(x1), 0)))
            ys, xs, line = skimage.draw.line_aa(y0, x0, y1, x1)
            kymo[ys, xs, 0] = line * 255  # Red line
        return kymo

    def draw_particles(self):
        """Draw each detection of each particle onto the kymogram."""
        # A. Rescale and transform to RGB
        kymo = cats.extensions.draw.grayscale_to_rgb(skimage.exposure.rescale_intensity(self.pixels, in_range=self.intensity_scale_from_particles()))

        # B. Draw
        # detections = [(d['t'], d['x']) for p in self.particles for d in p]
        for particle in self.particles:
            color = cats.utils.colors.random()
            for detection in particle:
                kymo[int(round(detection['x'], 0)), detection['t']] = color  # Red line

        return kymo

    def save(self, f=None):
        """Save the kymogram and its attributes into file f for future usage."""
        if f is None:
            f = self.file
        else:
            self.file = f
        with open(f, 'wb') as f:
            pickle.dump(self, f)

    def load(f):
        """Load a kymogram object from file f."""
        with open(f, 'rb') as f:
            return pickle.load(f)


def get_from(dataset, coordinates):
    """Return a list of Kymogram objects with their pixels defined.

    Faster than running a for loop on each coordinate.

    Parameters
    ----------
    dataset: pims object
        the dataset to extract the kymogram from
    coordinates: list of 2-tuples of 2-tuple
        a list containing tuples containing the positions (x, y) of the start and end of the kymogram in the dataset.
        [
        kymo1 ((x0, y0), (x1, y1)),
        kymo2 ((x0, y0), (x1, y1)),
        ...
        ]

    Returns
    -------
        kymograms: list
            list of Kymogram objects with their pixels defined.

    """
    # A. Build empty kymograms
    kymograms = list()
    for coords in coordinates:
        kymo = Kymogram(dataset, *coords)
        kymo.pixels = np.zeros((len(kymo.pixel_positions[0]), len(dataset)), dtype=np.uint16)
        kymograms.append(kymo)

    # B. Get the kymograms
    # Build frame by frame, and not kymogram by kymogram, to save time on reading images
    for i, frame in enumerate(dataset):
        for j, kymo in enumerate(kymograms):
            ys, xs = kymo.pixel_positions
            kymograms[j].pixels[:, i] = frame[ys, xs]

    return kymograms


def xml_to_coordinates(xml_file):
    """Transform a XML ROI list from Icy into a list of kymogram coordinates."""
    rois = list()
    for roi in ET.parse(xml_file).getroot().findall('roi'):
        pt1 = int(round(float(roi.find('pt1').find('pos_x').text), 0)), int(round(float(roi.find('pt1').find('pos_y').text), 0))
        pt2 = int(round(float(roi.find('pt2').find('pos_x').text), 0)), int(round(float(roi.find('pt2').find('pos_y').text), 0))
        rois.append((pt1, pt2))
    return rois


def csv_to_coordinates(csv_file):
    """Transform a CSV lines list from ImageJ into a list of kymogram coordinates.

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


def load_from_xml(data_dir, xml_file):
    """Transform a XML ROI list into a list of kymograms.

    Parameters:
    ----------
    data_dir: str
        The path to the directory containing the data
    xml_file: str
        The path to the XML file, as obtained from Icy, containing the position of the kymograms.

    """
    return get_from(pims.open(data_dir), xml_to_coordinates(xml_file))


def hand_picked_files_to_df(path):
    """Transform your loosy .csv files from ImageJ into a magnificient pandas DataFrame."""
    particles = list()
    for i, f in enumerate(glob(path)):
        values = pd.read_csv(f, header=None)
        values.rename(columns={0: 'x', 1: 'y'}, inplace=True)
        values['id'] = i
        particles.append(values)
    return pd.concat(particles) if len(particles) > 0 else pd.DataFrame([], columns=['x', 'y', 'id'])
