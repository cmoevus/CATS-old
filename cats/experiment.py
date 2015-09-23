#!env /usr/bin/python2
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# numpy's record arrays do not work with future unicode_literals
# from __future__ import unicode_literals

from .adict import dadict
from .sources import Images, ROI, globify
from . import defaults
from . import detection
from . import linkage
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gaussian_filter
from skimage.transform import rotate
from skimage import io, draw, exposure
from time import time
import networkx as nx
from colorsys import hls_to_rgb
from random import randrange
from math import ceil, atan, degrees
from multiprocessing import Pool
from sys import stdout
import pickle
import xml.etree.ElementTree as ET
import importlib
__all__ = ['Dataset', 'Experiment']


def analysis__methods(Class):
    """Decorate given class with functions from the analysis module."""
    analysis = importlib.import_module('CATS.analysis')
    for func in analysis.__all__:
        setattr(Class, func, getattr(analysis, func))
    return Class


class Dataset(Parameters):

    """
    A dataset is a collection of data that have the same experimental condition.

    Notably, these conditions include:
        - source of images
            - x, y, t offset
            - channel
        - experimental settings

    Datasets can take unlimited, user-defined parameters, including:
        - source: path, including wildcards for selecting specific files or BioFormat compatible file containing the images, or ROI/Images object.
            - x: the region of interest' x location. [first, last), in pixels. Default: all
            - y: the region of interest' y location. [first, last), in pixels. Default: all
            - t: the frames of interest. [first, last), in frames. Default: all
            - channel: the channel of the image. Default: 0
        - max_processes: the maximum number of processes to spawn when multiprocessing. Default: number of CPUs.

    To set or modify a parameter, simply call it as an attribute or as an item of the dataset, or during instantiation. Examples:
        D = Dataset('file.nd2', comment = 'This works, but only if the source is the first argument')
        D = Dataset(source = 'file.tif')
        D = Dataset({source: 'file.tif'}, channel = 0)
        D = Dataset(('source', 'file.tif'), channel = 0)
        D.source = 'dir_of_data/*.tif'
        D['source'] = '*c1*.jpg'
            ...
    Any pair of (parameter, value), or list of pairs, or dictionary, or object with a getitems() function, can be passed during instantiation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the object."""
        super(Dataset, self).__init__()
        self._defaults = defaults.Experiment
        self.spots = None
        self.tracks = None
        # If the first argument is a string, it is the source
        if len(args) > 0 and (type(args[0]) is str or isinstance(args[0], Images)):
            args = list(args)
            self.source = args.pop(0)
        self.update(*args, **kwargs)

    @property
    def source(self):
        """Return the location of the images (ROI object)."""
        return self.__getprop__('source')

    @source.setter
    def source(self, source):
        """
        Initialize the object representing the source of images.

        Dataset takes a ROI object as a source of images, but the user can provide:
            - an Images object
            - a ROI object
            - a path
        If a Images object or a path is provided, it will be used to initialize a ROI object.
        """
        if isinstance(source, ROI) == True:
            self.__setprop__('source', source)
        else:
            self.__setprop__('source', ROI(source))

    def save(self, f=None):
        """
        Save the dataset in its current form.

        A Dataset saved with save() and loaded with load() should be
        in the same state as left the last time it was saved.

        Arguments:
            f: file to save the dataset to. If None, the function returns the content of the would-be file
        """
        f = self.file if f is None and 'file' in self else f
        content = pickle.dumps(self)
        if f is not None:
            with open(f, 'w') as data:
                data.write(content)
            self.file = f
        return content if f is None else None

    def load(self, f=None):
        """
        Load a Dataset from a file or a string.

        A Dataset saved with save() and loaded with load() should be
        in the same state as left the last time it was saved.

        Arguments:
            f: file or string to load the dataset from. If None, will try to load from file in self.file.
        """
        f = self.file if f is None and 'file' in self else f
        if os.path.isfile(f) is True:
            with open(f, 'r') as data:
                D = pickle.load(data)
            self.file = f
        else:
            D = pickle.loads(f)
        self.update([i for i in D.__dict__.iteritems() if i[0] != '_attribs'], D)

    def test_detection_conditions(self, frame=0, output=None, save=True, **kwargs):
        """
        Test conditions for spots detection.

        Return a RGB image with a red pixel on each detected Gaussian center.

        Arguments
            frame: the frame to use, from the images, to test the conditions
            output: output file. If None, opens a window with the marked image
            save: whether to save (True) or not (False) the given detection conditions as Dataset's defaults.
        """
        D = Dataset(self.source.images)
        D.source.x, D.source.y, D.source.t = self.source.x, self.source.y, (frame, frame + 1)
        D.detection = kwargs
        D.detect_spots()
        images = D.draw(output=None)
        if output is None:
            io.imshow(images[0], cmap=plt.cm.gray)
            io.show()
        else:
            io.imsave(output, images[0])
        if save == True:
            self.detection = kwargs
        return D.spots

    def detect_spots(self, verbose=True):
        """
        Find the blobs on the Dataset's images.

        Uses values from Dataset.detection, that can be set using
        test_detection_conditions()
        The number of simultaneous processes can be modified using
        Dataset.max_processes, which defaults to the number of CPUs
        (max speed, at the cost of slowing down all other work)

        Arguments
            verbose: Writes about the time and frames and stuff as it works
        """
        # Get the detection function
        detection_func = getattr(detection, self.detection.method)

        # Multiprocess through it
        t = time()
        spots, pool = list(), Pool(self.max_processes)
        for blobs in pool.imap(detection_func, iter((i, j, self.detection) for j, i in enumerate(self.source.read()))):
            if verbose is True:
                print('\rFound {0} spots in frame {1}. Process started {2:.2f}s ago.         '.format(len(blobs), blobs[0][4] if len(blobs) > 0 else 'i', time() - t), end='')
                stdout.flush()
            spots.extend(blobs)
        pool.close()
        pool.join()

        # Save the spot list into memory
        if verbose is True:
            print('\nFound {0} spots in {1} frames in {2:.2f}s'.format(len(spots), self.source.length, time() - t))
        self.spots = np.array(spots, dtype={'names': ('x', 'y', 's', 'i', 't'), 'formats': (float, float, float, float, int)}).view(np.recarray)
        return spots

    def link_spots(self, verbose=True):
        """
        Link spots together in time.

        Argument:
            verbose: Writes about the time and frames and stuff as it works
        """
        # Get the detection function
        linkage_func = getattr(linkage, self.linkage.method)
        self.tracks = linkage_func(self, verbose)

    def filter(self, overwrite=True, parameters=None):
        """
        Filter the tracks obtained by link_spots based on parameters set either in self.filtration, either passed as arguments.

        Argument
            overwrite: bool. If True, the Dataset's tracks will be replaced by the filtered ones.
            parameters: dict. Parameters to use for filtering. If None, will use the ones from the Dataset's parameters (Dataset.filtration).
        """
        if parameters is None:
            parameters = self.filtration
        if parameters['max_length'] is None:
            parameters['max_length'] = self.source.length
        n_tracks = list()
        for track in self.tracks:
            if parameters['min_length'] <= len(track) <= parameters['max_length']:
                if np.abs(np.diff(sorted([self.spots[s]['t'] for s in track]))).mean() <= parameters['mean_blink']:
                    n_tracks.append(track)

        if overwrite is True:
            self.filtration = parameters
            self.tracks = n_tracks

        return n_tracks

    def draw(self, output=None, spots=None, tracks=True, rescale=True):
        """
        Draw the tracks and/or spots onto the images of the dataset as RGB images.

        Argument:
            output: the directory in which to write the files. If None, returns the images as a list of arrays.
            spots: if True, surrounds the spots on the images with a red circle. If None, only show spots if no tracks are available.
            tracks: if present, surround each of the spots in the tracks on the images with a colored circle. All spots within the same track will have the same color.
            rescale: adjust intensity levels to the detected content
        """
        if output is None:
            images = list()

        if spots is None and self.tracks is None:
            spots = True

        # Separate tracks per image
        if tracks is True and self.tracks is not None:
            # Make colors
            colors = [np.array(hls_to_rgb(randrange(0, 360) / 360, randrange(20, 80, 1) / 100, randrange(20, 80, 10) / 100)) * 255 for i in self.tracks]
            frames = [[] for f in range(self.source.length)]
            for track, color in zip(self.get_tracks(), colors):
                sigma = int(np.mean([s['s'] for s in track]))
                for s in track:
                    spot = (int(s['y']), int(s['x']), sigma, color)
                    frames[s['t']].append(spot)

        # Calculate intensity range:
        if rescale == True:
            if tracks is True and self.tracks is not None:
                i = [i for t in self.get_tracks('i') for i in t]
            elif spots is True and self.spots is not None:
                i = [s['i'] for s in self.spots]
            if len(i) != 0:
                m, s = np.mean(i), np.std(i)
                scale = (m - 3 * s, m + 3 * s)
            else:
                scale = False

        shape = self.source.dimensions[::-1]
        for t, image in enumerate(self.source.read()):
            # Prepare the output image (a 8bits RGB image)
            ni = np.zeros([3] + list(self.source.dimensions[::-1]), dtype=np.uint8)

            if rescale is True and scale is not False:
                image = exposure.rescale_intensity(image, in_range=scale)

            ni[..., :] = image / image.max() * 255
            ni = ni.transpose(1, 2, 0)

            # Mark spots
            if spots is True and self.spots is not None:
                for s in (self.spots[i] for i in np.where(self.spots['t'] == t)[0]):
                    area = draw.circle_perimeter(int(s['y']), int(s['x']), int(s['s']), shape=shape)
                    ni[area[0], area[1], :] = (255, 0, 0)

            # Mark tracks
            if tracks is True and self.tracks is not None:
                for s in frames[t]:
                    area = draw.circle_perimeter(s[0], s[1], s[2], shape=shape)
                    ni[area[0], area[1], :] = s[3]

            # Show/save image
            if output is None:
                images.append(ni)
            else:
                io.imsave("{0}/{1}.tif".format(output, t), ni)
        return images if output is None else None

    def get_tracks(self, properties=None):
        """
        Return the tracks in the Dataset as a list of spots, rather than a list of spot ids.

        Arguments:
            properties: list of the properties to return for each spots.

        Will return the properties in the same order as given. If None,
        returns all properties.
            - x: the position, in x
            - y: the position, in y
            - s: the standard deviation of the gaussian kernel
            - i: the intensity at (x, y)
            - t: the frame
            Example list: ['x', 'y', 't']
        """
        spots = self.spots[properties] if properties is not None else self.spots
        tracks = list()
        for track in self.tracks:
            t = list()
            for s in track:
                t.append(spots[s])
            tracks.append(t)
        return tracks

    def export_track(self, track, properties=None):
        """
        Return the given track as a list of spot properties rather than as a list of spot ids.

        Arguments:
            track: a list of spot ids from the dataset
            properties: list of the properties to return for each spots.

        Will return the properties in the same order as given. If None,
        returns all properties.
            - x: the position, in x
            - y: the position, in y
            - s: the standard deviation of the gaussian kernel
            - i: the intensity at (x, y)
            - t: the frame
        """
        spots = self.spots[properties] if properties is not None else self.spots
        t = list()
        for s in track:
            t.append(spots[s])
        return t

    def as_isbi_xml(self, f='result.xml', snr=7, density='low med high 100.0', scenario='NO_SCENARIO'):
        """Return tracks as ISBI 2012 Challenge XML for scoring."""
        root = ET.Element('root')
        t = ET.SubElement(root, 'TrackContestISBI2012')
        t.attrib = {'SNR': str(snr), 'density': density, 'scenario': scenario}
        for track in self.get_tracks():
            p = ET.SubElement(t, 'particle')
            for spot in track:
                s = ET.SubElement(p, 'detection')
                s.attrib = {'t': str(spot['t']),
                            'x': str(round(spot['x'], 2)),
                            'y': str(round(spot['y'], 2)),
                            'z': '0'}
        E = ET.ElementTree(element=root)
        E.write(f)


@analysis__methods
class Experiment(Parameters):

    """
    Representation of a single-molecule experiment.

    An experiment can be:
        1) A set of data collected in some specific conditions
        2) A collection of datasets aiming to test an hypothesis
    This class reflects both definitions. An Experiment can contain several sources of images, from which it will extract datasets. Data from each dataset will be merged as the data of the Experiment.

    To initiate an Experiment, you can use:
        The path (or a list of path) to the images, plus keywords arguments, if desired
        A path to a save file.
        Any combination of keywords/list of parameters/dicts, just like for Datasets

    Properties:

    sources: the experiment's images. Can set it as a path or a list of paths.
    source: same as sources, but will return None if no sources, or the object if there only is one source.
    """

    def __init__(self, *args, **kwargs):
        """Load given sources and arguments."""
        super(Parameters, self).__init__(_defaults=defaults.Experiment)
        self.__setprop__('sources', [])
        self.__setprop__('datasets', [])
        args = list(args)
        if len(args) > 0 and isinstance(args[0], str):
            try:
                self.load(args[0])
                del args[0]
            except:
                self.add_dataset(args[0])
                del[args[0]]
        self.update(*args, **kwargs)

    @property
    def sources(self):
        """List Images objects associated with the Experiment."""
        return self.__getprop__('sources')

    @sources.setter
    def sources(self, value):
        """Add/Set Images object(s) to the Experiment."""
        self.__setprop__('sources', list())
        if '__iter__' not in dir(value):
            value = [value]
        for v in value:
            self.add_source(v)

    @property
    def source(self):
        """Dumb unary alias for sources."""
        sources = self.__getprop__('sources')
        if len(sources) == 0:
            return None
        elif len(sources) == 1:
            return sources[0]
        else:
            return sources

    @source.setter
    def source(self, value):
        """Dumb unary alias for sources."""
        return setattr(self, 'sources', value)

    def add_source(self, source):
        """
        Add a source to the existing list.

        Argument:
            source: a path or an Images object.
        """
        if self.has_source(source) == False:
            if isinstance(source, Images):
                self.sources.append(source)
            else:
                self.sources.append(Images(source))

    def has_source(self, source):
        """
        Return the source if it is in the Experiment has the given source, return False if not.

        Argument:
            source: Images, ROI or string representing a source of images (see Images)
        """
        source = source.source if isinstance(source, Images) else globify(source)
        sources = [s.source for s in self.sources]
        if source in sources:
            return self.sources[sources.index(source)]
        else:
            return False

    @property
    def datasets(self):
        """List the datasets associated with the Experiment."""
        return self.__getprop__('datasets')

    @datasets.setter
    def datasets(self, value):
        """Set/Add datasets to the Experiment."""
        self.__setprop__('datasets', list())
        if isinstance(value, Dataset) or '__iter__' not in dir(value):
            # Because Dataset also has a __iter__ method...
            value = [value]
        for v in value:
            self.add_dataset(v)

    def add_dataset(self, dataset):
        """
        Add a dataset to the existing list.

        To avoid duplication of Images objects, the Images objects will be centralize so that (1) the given datasets' source is changed to the object already in the Experiment, if they represent the same files or (2) the datasets' source is added to the Experiment.

        Argument:
            dataset: anything that can initiate a Dataset object.
        """
        if isinstance(dataset, Images) or isinstance(dataset, str):
            source = self.has_source(dataset)
            if source == False:
                if isinstance(dataset, ROI):
                    self.add_source(dataset.images)
                else:
                    self.add_source(dataset)
                source = self.sources[-1]
            if isinstance(dataset, ROI):
                dataset.images = source
                dataset = Dataset(dataset, _defaults=self)
            else:
                dataset = Dataset(source, _defaults=self)
        else:
            if isinstance(dataset, Dataset) == False:
                dataset = Dataset(dataset)
            source = self.has_source(dataset.source)
            if source == False:
                self.add_source(dataset.source.images)
            else:
                dataset.source.images = source
            dataset._defaults = self
        self.datasets.append(dataset)

    def save(self, f=None):
        """
        Save the Experiment in its current state for future use.

        An Experiment saved with save() and loaded with load() should be in the same state as left the last time it was saved.

        Arguments:
            f: file to save the experiment to. If None, the experiment is saved to the last file it was saved to or loaded from, which is written in the 'file' parameter of the experiment. If this parameter does not exist, the function returns the content of the would-be file as an array of pickled objects.
        """
        f = self.file if f is None and 'file' in self else f
        content = pickle.dumps(self)
        if f is not None:
            with open(f, 'w') as data:
                data.write(content)
            self.file = os.path.abspath(f)
        return content if f is None else None

    def load(self, f=None):
        """
        Load an Experiment from a file or a string.

        An Experiment saved with save() and loaded with load() should be in the same state as left the last time it was saved.

        Arguments:
            f: file or array of pickled objects to load the experiment from. If None, the experiment is loaded from the last file itwas saved to or loaded from, which is written in the 'file' parameter of the experiment.
        """
        f = self.file if f is None and 'file' in self else f
        if os.path.isfile(f) is True:
            with open(f, 'r') as data:
                E = pickle.load(data)
            self.file = os.path.abspath(f)
        else:
            E = pickle.loads(f)
        self.__setprop__('sources', E.sources)
        self.__setprop__('datasets', E.datasets)
        self.update([i for i in E.__dict__.iteritems() if i[0] not in ('_attribs', 'sources', 'datasets')], E)

    def detect_spots(self, verbose=True):
        """Detect spots in each of the underlying datasets."""
        t, n = time(), 0
        for i, ds in enumerate(self.datasets):
            if verbose is True:
                print('\rWorking on dataset {0}'.format(i), end='\n')
                stdout.flush()
            ds.detect_spots(verbose)
            n += len(ds.spots)
        if verbose is True:
            print('\rFound {0} spots in {1} datasets in {2:.2f}s:\n'.format(n, i + 1, time() - t), end='')
            stdout.flush()

    def link_spots(self, verbose=True):
        """Link spots to form tracks in each of the underlying datasets."""
        t, n = time(), 0
        for i, ds in enumerate(self.datasets):
            if verbose is True:
                print('\rWorking on dataset {0}'.format(i), end='\n')
                stdout.flush()
            ds.link_spots(verbose)
            n += len(ds.tracks)
        if verbose is True:
            print('\rFound {0} tracks in {1} datasets in {2:.2f}s:\n'.format(n, i + 1, time() - t), end='')
            stdout.flush()

    def filter(self, overwrite=True, parameters=None):
        """Filter tracks in each of the underlying datasets."""
        filtered_tracks = list()
        for ds in self.datasets:
            tracks = ds.filter(overwrite, parameters)
            for track in tracks:
                filtered_tracks.append(ds.export_track(track))
        return filtered_tracks

    def get_tracks(self, properties=None):
        """
        Return the tracks from each dataset of the experiment as a list of spot properties.

        Arguments:
            properties: list of the properties to return for each spots.

        Will return the properties in the same order as given. If None, returns all properties.
            x: the position, in x
            y: the position, in y
            s: the standard deviation of the gaussian kernel
            i: the intensity at (x, y)
            t: the frame
        Example list: ['x', 'y', 't']
        """
        tracks = list()
        for ds in self.datasets:
            tracks.extend(ds.get_tracks(properties))
        return tracks

    def find_barriers(self, datasets=None, overwrite=True):
        """
        Find the barriers in an experiment.

        NEED TO IMPLEMENT:
            - IMCOMPLETE BARRIER SETS

        Arguments:
            datasets: list of the ids of the datasets to find barriers in. If None, tries all datasets
            overwrite: bool. If True, the given datasets will be replaced by one dataset for each set of their barriers.
        """
        barrier_datasets = list()
        if datasets is None:
            datasets = range(len(self.datasets))
        for ds in datasets:
            ds = self.datasets[ds]
            dbp, dbb = self.barriers.dbp, self.barriers.dbb
            approx = self.barrier_detection.approx

            # Make a projection of the image
            d_tlims = ds.barrier_detection.tlims
            ds.images.tlims = d_tlims if d_tlims is not None else ds.tlims
            img = np.zeros(ds.images.dimensions()[::-1])
            for i in ds.images.read():
                img += i
            ds.images.tlims = ds.tlims
            img = gaussian_filter(img, self.barrier_detection.blur)
            axis = 0 if self.barriers.axis == 'y' else 1
            projection = np.sum(img, axis=axis)

            # Find peaks
            d_proj = np.diff(projection)
            tangents = list()
            for i in range(0, len(d_proj) - 1):
                neighbors = sorted((d_proj[i], d_proj[i + 1]))
                if neighbors[0] < 0 and neighbors[1] > 0:
                    tangents.extend([i])
            extremas = np.where(np.square(np.diff(d_proj)) - np.square(d_proj[:-1]) >= 0)[0]
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
            if 'barriers' not in self:
                self.barriers = {}
            self.barriers.positions = sorted([sorted(n[0]) for n in sets[set_scores.index(sorted(set_scores, reverse=True)[0])]])

            # Transform barrier sets into datasets
            for barrier_set in self.barriers.positions:
                b_ds = Dataset()
                b_ds.update(ds)
                lims = 'xlims' if self.barriers.axis == 'y' else 'ylims'
                if self.barriers.orientation == 'pb':
                    barrier_set = barrier_set[::-1] + [-1]
                setattr(b_ds, lims, barrier_set)
                barrier_datasets.append(b_ds)

        # Overwrite previous datasets if required
        if overwrite is True:
            for ds in datasets:
                del self.datasets[ds]
            self.datasets.extend(barrier_datasets)

        return barrier_datasets


class OldExperiment(object):

    def find_barriers(self, datasets=None, overwrite=True):
        """
        Find the barriers in an experiment.

        NEED TO IMPLEMENT:
            - IMCOMPLETE BARRIER SETS

        Arguments:
            datasets: list of the ids of the datasets to find barriers in. If None, tries all datasets
            overwrite: bool. If True, the given datasets will be replaced by one dataset for each set of their barriers.
        """
        barrier_datasets = list()
        if datasets is None:
            datasets = range(len(self.datasets))
        for ds in datasets:
            ds = self.datasets[ds]
            dbp, dbb = self.barriers.dbp, self.barriers.dbb
            approx = self.barrier_detection.approx

            # Make a projection of the image
            d_tlims = ds.barrier_detection.tlims
            ds.images.tlims = d_tlims if d_tlims is not None else ds.tlims
            img = np.zeros(ds.images.dimensions()[::-1])
            for i in ds.images.read():
                img += i
            ds.images.tlims = ds.tlims
            img = gaussian_filter(img, self.barrier_detection.blur)
            axis = 0 if self.barriers.axis == 'y' else 1
            projection = np.sum(img, axis=axis)

            # Find peaks
            d_proj = np.diff(projection)
            tangents = list()
            for i in range(0, len(d_proj) - 1):
                neighbors = sorted((d_proj[i], d_proj[i + 1]))
                if neighbors[0] < 0 and neighbors[1] > 0:
                    tangents.extend([i])
            extremas = np.where(np.square(np.diff(d_proj)) - np.square(d_proj[:-1]) >= 0)[0]
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
            if 'barriers' not in self:
                self.barriers = {}
            self.barriers.positions = sorted([sorted(n[0]) for n in sets[set_scores.index(sorted(set_scores, reverse=True)[0])]])

            # Transform barrier sets into datasets
            for barrier_set in self.barriers.positions:
                b_ds = Dataset()
                b_ds.update(ds)
                lims = 'xlims' if self.barriers.axis == 'y' else 'ylims'
                if self.barriers.orientation == 'pb':
                    barrier_set = barrier_set[::-1] + [-1]
                setattr(b_ds, lims, barrier_set)
                barrier_datasets.append(b_ds)

        # Overwrite previous datasets if required
        if overwrite is True:
            for ds in datasets:
                del self.datasets[ds]
            self.datasets.extend(barrier_datasets)

        return barrier_datasets

    def correct_rotation(self, source, destination, bins=10):  # NOT TRANSFERED TO OOP
        """
        Corrects rotation on images based on the barriers
        Assumes that rotation is constant over time.

        source: the directory containing the images
        destination: where to write the corrected images
        bins: number of bins to split the image in, vertically
              (lower if noise is higher)
        """
        images = sorted(os.listdir(source))
        shape = io.imread(source + '/' + images[0]).shape
        bins = np.linspace(0, shape[1], bins, dtype=int)
        barriers = list()
        for i in range(len(bins) - 1):
            barriers.append([i for j in self.find_barriers(source, ylims=(bins[i], bins[i + 1])) for i in j])

        m = list()
        for y in np.array(barriers).T:
            y -= y[0]
            m.append(np.polyfit(bins[1:], y, 1)[0])
        angle = degrees(atan(np.mean(m)))

        for i in images:
            io.imsave(destination + '/' + i, rotate(io.imread(source + '/' + i), angle,
                      mode='nearest',
                      preserve_range=True).astype(dtype=np.int16))

        return True


def transitive_reduction(G, order=None, adj_matrix=False):
    """
    Return the transitive reduction of a given graph.

    Based on Aho, A. V., Garey, M. R., Ullman, J. D. The Transitive
    Reduction of a Directed Graph. SIAM Journal on Computing 1, 131–137
    (1972).

    Arguments:
        order: the order in which the vertices appear in time (to fit the second condition). If None, uses the same order as in the graph
        adj_matrix: returns an adjacency matrix if True, a Graph if False
    """
    # Transitive closure
    MT = transitive_closure(G, order, True)

    # Reorganize the adjacency matrix
    M = nx.to_numpy_matrix(G, nodelist=order, weight=None)

    # Adjacency matrices operations
    Mt = np.array(M - M.dot(MT))

    # Return in the proper format
    if adj_matrix is True:
        return np.where(Mt != 1, 0, 1).astype(dtype=bool)
    else:
        Gt = G.copy()
        for i, j in np.vstack(np.where(Mt != np.array(M))).T:
            try:
                Gt.remove_edge(order[i], order[j])
            except:
                pass
        return Gt


def transitive_closure(G, order=None, adj_matrix=False):
    """
    Return the transivite closure of a graph.

    This method assumes that your graph is:
        1. Directed acyclic
        2. Organized so that a node can only interact with a node
        positioned after it, in the adjacency matrix

    Arguments:
        order: the order in which the vertices appear in time (to fit the second condition)
        adj_matrix: returns an adjacency matrix if True, a Graph if False
    """
    M = nx.to_numpy_matrix(G, nodelist=order, weight=None)

    # Close the graph
    for i, j in sorted((np.array(np.where(M == 1), dtype=int).T)[0], key=lambda a: [0])[::-1]:
        M[i] = np.logical_or(M[i], M[j])

    # Return in the proper format
    if adj_matrix is True:
        return M
    else:
        GT = G.copy()
        for i, j in (np.array(np.where(M == 1)).T):
            GT.add_edge(order[i], order[j])
        return GT
