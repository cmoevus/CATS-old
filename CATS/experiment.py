#!env /usr/bin/python2
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# numpy's record arrays do not work with future unicode_literals
# from __future__ import unicode_literals

from .parameters import Parameters
from .sources import Images, ROI, globify
from . import defaults
import os
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage.feature import blob_log
from skimage.filters import gaussian_filter
from skimage.transform import rotate
from skimage import io, draw, exposure
from time import time
import networkx as nx
from colorsys import hls_to_rgb
from random import randrange
from math import floor, ceil, atan, degrees
from multiprocessing import Pool
from sys import stdout
import pickle
import tarfile
from StringIO import StringIO
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
        # Multiprocess through it
        t = time()
        spots, pool = list(), Pool(self.max_processes)
        for blobs in pool.imap(find_blobs, iter((i, self.detection.blur, self.detection.threshold, j) for j, i in enumerate(self.source.read()))):
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
        # Reorganize spots by frame and prepare the Graph
        G = nx.DiGraph()
        n_frames = self.source.length
        frames = [[] for f in range(n_frames)]
        for i, spot in enumerate(self.spots):
            frames[spot['t']].append((spot['x'], spot['y'], i))
            G.add_node(i, frame=spot['t'])

        # Make optimal pairs for all acceptable frame intervals
        # (as defined in max_blink)
        for delta in range(1, self.linkage.max_blink + 1):
            if verbose is True:
                print('\rDelta frames: {0}'.format(delta), end='')
                stdout.flush()
            for f in range(n_frames - delta):
                # Matrix of distances between spots
                d = np.abs(np.array(frames[f])[:, np.newaxis, :2] -
                           np.array(frames[f + delta])[:, :2])

                # Filter out the spots with distances that excess max_disp in x and/or y
                disp_filter = d - self.linkage.max_disp >= 0
                disp_mask = np.logical_or(disp_filter[:, :, 0], disp_filter[:, :, 1])

                # Reduce to one dimension
                d = np.sqrt(d[:, :, 0]**2 + d[:, :, 1]**2)

                # Find the optimal pairs
                f_best = d != np.min(d, axis=0)
                fd_best = (d.T != np.min(d, axis=1)).T
                mask = np.logical_or(f_best, fd_best)
                mask = np.logical_or(mask, disp_mask)
                pairs = ma.array(d, mask=mask)

                # Organize in pairs, or edges, for graph purposes
                for s1, s2 in np.array(np.where(pairs.mask == False)).T:
                    G.add_edge(frames[f][s1][2], frames[f + delta][s2][2],
                               weight=d[s1][s2])

        # Only keep the tracks that are not ambiguous (1 spot per frame max)
        tracks, a_tracks = list(), list()
        for track in nx.weakly_connected_component_subgraphs(G):
            t_frames = np.array(sorted([(s, self.spots[s]['t']) for s in track], key=lambda a: a[1]))

            # Good tracks
            if len(t_frames[:, 1]) == len(set(t_frames[:, 1])):
                tracks.append(sorted(track.nodes(), key=lambda s: self.spots[s]['t']))

            # Ambiguous tracks
            # This is a work in progress. More or less abandoned.
            elif self.linkage.ambiguous_tracks == True:
                nodes = track.nodes()
                track = dict([(self.spots[s]['t'], []) for s in nodes])
                for s in nodes:
                    track[self.spots[s]['t']].append(s)
                ts = sorted(track.keys())
                for t in ts[:-1]:
                    if len(track[t]) > 1:
                        now = track[t]
                        t_after = ts.index(t) + 1
                        after = track[ts[t_after]]
                        scores = np.abs(np.array([[self.spots[s]['x'], self.spots[s]['y']] for s in now])[:, np.newaxis] - np.array([[self.spots[s]['x'], self.spots[s]['y']] for s in after]))
                        scores = scores[:, :, 0] + scores[:, :, 1]
                        pair = np.where(scores == scores.min())
                        # print([self.spots[s] for s in now], [self.spots[s] for s in after], pair)
                        if len(pair[0]) > 1:
                            pair = (np.array(pair).T)[0]
                        now, after = [now[pair[0]]], [after[pair[1]]]
                track = sorted([t[0] for t in track.values()], key=lambda a: self.spots[a]['t'])
                ts = [self.spots[s]['t'] for s in track]
                a_tracks.append(track)

        if verbose is True:
            print('\nFound {0} tracks'.format(len(tracks)))

        if self.linkage.ambiguous_tracks == False:
            self.tracks = tracks
            return tracks
        else:
            self.tracks = tracks + a_tracks
            return tracks, a_tracks

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

    """
    An experiment can be:
        1) A set of data collected in some specific conditions
        2) A collection of datasets aiming to test an hypothesis
    This class reflects both definitions

    Set arguments directly through Experiment.argument = value. Do not
    use Experiment.parameters['argument'] = value, as this will not
    transfer the arguments to the underlying Datasets.

    To initiate an Experiment, you can use:
    - The path (or a list of path) to the images, plus keywords
    arguments, if desired
    or
    - Any combination of keywords/list of parameters/dicts, just like
    for Datasets
    """

    def __init__(self, *args, **kwargs):
        self.datasets = []
        self.parameters = Parameters()

        # Only one argument given: path to the the save file or images
        if len(args) == 1:
            try:
                load = tarfile.is_tarfile(args[0])
            except:
                load = False
            if load is True:
                self.load(args[0])
            else:
                self.source = args[0]

        # Import dicts and lists and pairs
        else:
            for d in args:
                if 'iteritems' in dir(d) or isinstance(d, Parameters):
                    d = d.iteritems()
                elif '__iter__' not in dir(d[0]):
                    d = (d, )
                for k, v in d:
                    setattr(self, k, v)

        # Import keywords
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def __getattr__(self, attr):
        try:
            return self.parameters[attr]
        except KeyError:
            raise AttributeError('{0} is not set in this Experiment'.format(attr))

    def __setattr__(self, attr, value):
        """
        Special handling for parameters
        The Experiment class passes down all given parameters to its
        Datasets, for the best and for the worst. With great powers
        come great responsibilities.
        """
        if attr in dir(self):
            object.__setattr__(self, attr, value)
        else:
            for ds in self.datasets:
                setattr(ds, attr, value)
            self.parameters[attr] = value

    def __contains__(self, attr):
        s = self.__dict__.__contains__(attr)
        return s if s is True else self.__dict__['parameters'].__contains__(attr)

    @property
    def parameters(self):
        return self.__dict__['parameters']

    @parameters.setter
    def parameters(self, value):
        p = Parameters(value)
        self.__dict__['parameters'] = Parameters()
        for n, v in p.iteritems():
            setattr(self, n, v)

    @property
    def datasets(self):
        return self.__dict__['datasets']

    @datasets.setter
    def datasets(self, value):
        self.__dict__['datasets'] = value

    @property
    def source(self):
        s = set()
        for ds in self.datasets:
            s.add(ds.source)
        l = len(s)
        if l == 1:
            return s.pop()
        elif l == 0:
            return None
        else:
            return s

    @source.setter
    def source(self, source):
        if len(self.datasets) == 0:
            if '__iter__' not in dir(source):
                source = (source, )
            for s in source:
                self.add_dataset(s)
        else:
            raise AttributeError("Cannot set a source on an Experiment that is"
                                 "already initialized. Use "
                                 "Experiment.add_dataset() instead.")

    def add_dataset(self, *args, **kwargs):
        """
        Add a dataset to the experiment.

        Possible arguments:
        ------------------
        A) Dataset object. The dataset will directly be added to the Experiment.
        B) list/dict of parameters or keywords: the Dataset will receive the Experiment's parameters first, and then those provided
        C) string (Path to the source of images) + list/keywords parameters. The Dataset source will be set with the string (has to be the first argument). Keywords or lists/dicts can be added as parameters, that will be added on top of the Experiment's parameters.
        """
        args = list(args)
        if len(args) == 1 and isinstance(args[0], Dataset):
            self.datasets.append(args[0])
        else:
            d = Dataset(self.parameters)
            if len(args) > 0 and type(args[0]) == str:
                d.source = args.pop()
            d.update(*args, **kwargs)
            self.datasets.append(d)

    def save(self, f=None):
        """
        Save the Experiment in its current state for future use.

        An Experiment saved with save() and loaded with load() should be in the same state as left the last time it was saved.

        Arguments:
            f: file to save the experiment to. If None, the experiment is saved to the last file it was saved to or loaded from, which is written in the 'file' parameter of the experiment. If this parameter does not exist, the function returns the content of the would-be file as an array of pickled objects.
        """
        content = [pickle.dumps(d) for d in self.datasets]
        content.append(pickle.dumps(self.parameters))
        f = self.file if f is None and 'file' in self else f
        if f is not None:
            with tarfile.open(f, 'w:bz2') as t:
                # Found solution to avoid making temp files there:
                # http://stackoverflow.com/q/740820
                for n, c in enumerate(content):
                    c = StringIO(c)
                    name = 'Dataset_{0}'.format(n) if n < len(self.datasets) else 'Parameters'
                    i = tarfile.TarInfo(name=name)
                    i.size = len(c.buf)
                    t.addfile(tarinfo=i, fileobj=c)
            self.file = os.path.abspath(f)
        return content if f is None else None

    def load(self, f=None):
        """
        Load an Experiment from a file or a string.

        An Experiment saved with save() and loaded with load() should be in the same state as left the last time it was saved.

        Arguments:
            f: file or array of pickled objects to load the experiment from. If None, the experiment is loaded from the last file itwas saved to or loaded from, which is written in the 'file' parameter of the experiment.
        """
        if f is None:
            if 'file' in self:
                f = self.file
            else:
                raise AttributeError('The experiment has no file associated'
                                     'and no input parameter was given.')
        try:
            if tarfile.is_tarfile(f) == True:
                setattr(self, 'file', os.path.abspath(f))
                with tarfile.open(f, 'r:bz2') as t:
                    content = [t.extractfile(i).read() for i in t.getmembers()]
        except TypeError:
            content = f
        self.datasets = []
        for n, p in enumerate(content):
            if n < len(content) - 1:
                self.add_dataset(pickle.loads(p))
            else:
                self.parameters = pickle.loads(p)
                if type(f) == str:
                    self.file = os.path.abspath(f)

    def test_detection_conditions(self):  # NOT IMPLEMENTED YET
        """Find the optimal detection conditions in each of the underlying dataset."""

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
    image, blur, threshold, extra = args[0], args[1], args[2], args[3:]
    b = blob_log(ndimage.gaussian_filter(image, blur), threshold=threshold, min_sigma=1)
    blobs = list()
    for y, x, s in b:
        blob = [x, y, s, image[y][x]]
        blob.extend(extra)
        blobs.append(tuple(blob))
    return fit_gaussian_on_blobs(image, blobs)


def fit_gaussian_on_blobs(*args):
    """
    Fit a Gaussian curve on the blobs in an image. Return the given list of blobs with the fitted values.

    Arguments (as a list of both, for multiprocessing):
        img: the ndarray of the image containing the blobs
        blobs: a list of blobs (x, y, sigma, intensity, ...) that need to go subpixel resolution. All extra information after 'intensity'  will be kept in the output.
    """
    if len(args) == 1:
        args = args[0]
    img, blobs = args[0], args[1]
    spr_blobs = list()
    for blob in blobs:
        try:
            r = blob[2] + 1 * np.sqrt(2)
            ylims, xlims = img.shape
            y = (max(0, int(floor(blob[1] - r))), min(int(ceil(blob[1] + r)) + 1, ylims))
            x = (max(0, int(floor(blob[0] - r))), min(int(ceil(blob[0] + r)) + 1, xlims))
            data = img[y[0]:y[1], x[0]:x[1]]
            coords = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))

            fit, cov = curve_fit(gaussian_2d, coords, data.ravel(), p0=(blob[3], blob[0] - x[0], blob[1] - y[0], blob[2]))

            spr_blob = [x[0] + fit[1], y[0] + fit[2], fit[3], fit[0]]
            if len(blob) > 4:
                spr_blob.extend(blob[4:])
            if x[0] <= spr_blob[0] <= x[1] and y[0] <= spr_blob[1] <= y[1]:
                spr_blobs.append(tuple(spr_blob))
            else:
                raise ValueError
        except:
            # spr_blobs.append(blob)
            pass

    return spr_blobs


def gaussian_2d(coords, A, x0, y0, s):
    """Draw a 2D gaussian with given properties."""
    x, y = coords
    return (A * np.exp(((x - x0)**2 + (y - y0)**2) / (-2 * s**2))).ravel()


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
