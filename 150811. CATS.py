#!env /usr/bin/python2
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals # Does not work with numpy's record arrays

import os
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
from scipy import ndimage, misc, signal, stats
from scipy.optimize import curve_fit
from skimage.feature import blob_log
from skimage.filters import threshold_otsu, gaussian_filter
from skimage.transform import rotate
from skimage import io
from time import time
import networkx as nx
from colorsys import hls_to_rgb
from random import randrange, randint
from math import floor, ceil, atan, degrees
from itertools import product
import sys
from glob import glob, iglob
import javabridge
import bioformats
from multiprocessing import Pool, cpu_count
from sys import stdout
import pickle
import tarfile
from StringIO import StringIO
import xml.etree.ElementTree as ET

class Parameters(object):
    '''
    Parameters dictionary-object. Allows to add getters and setters on
    parameters, as well as to access them through Parameters.property
    or Parameters['property'], which means it can still be passed as
    **kwargs

    Parameters starting with '_' will be hidden. These can be
    Parameters' parameters or voluntarly hidden parameters. They have
    to be explicitely called by the user. They will not be inherited
    or listed by update() or items()/keys()/values()/etc.
    _position: (list) position within the 'DEFAULTS' parameters
    _use_defaults: (bool) use the DEFAULTS parameters when a value is
    not found in the current Parameters object

    In order to keep keywords logical between the dictionary view and
    the attribute view of the object, when writing getters and setters,
    use self.__dict__['params'][k] = v, for example, rather than
    object.__setattr__(self, 'k', v). Indeed, the latter case will lead
    to infinite recursion. Using a different 'transition' name will
    lead to improper listing when iterating over the values, for example
    in **kwargs.

    Arguments:
    ---------
    Any dictionary, iterable object that can be accessed by iteritems()
    or as a pair of (keyword, value). Acceptable arguments are:
    - Dictionaries
    - Objects with a iteritems() function
    - Tuples/Lists of (keywork, value)
    - (keyword, value) as a tuple/list
    - keyworded arguments
    '''

    fetched_functs = ['keys', 'values', 'items', 'iterkeys', 'iteritems', 'itervalues', '__iter__']

    def __init__(self, *args, **kwargs):
        if '_position' in kwargs:
            self._position = kwargs['_position']
        else:
            self._position = []
        self.__dict__['params'] = dict()
        self.update(*args, **kwargs)
        if hasattr(self, '_use_defaults') is False:
            self._use_defaults = True

    def __getattr__(self, attr):
        '''
        fetched_functs allows to handle selected basic dictionary functions
        '''
        error = '{0} not found'.format(attr)
        if attr in self.fetched_functs:
            return getattr(self.__dict__['params'], attr)
        else:
            try:
                if attr[0] != '_':
                    return self.__dict__['params'][attr]
                else:
                    raise KeyError
            except KeyError:
                if self._use_defaults is False:
                    raise AttributeError(error)
                else:
                    try :
                        d = DEFAULTS
                        for p in self._position:
                            d = d[p]
                        if attr in d:
                            # Avoid modifying the DEFAULTS dict. Kind of a lame hack...
                            if isinstance(d[attr], Parameters):
                                return d[attr].copy()
                            else:
                                return d[attr]
                        else:
                            raise KeyError
                    except KeyError:
                        raise AttributeError(error)

    def __setattr__(self, attr, value):
        # Prevent from blocking the read-only attributes from __dict__
        if attr in self.fetched_functs:
            raise AttributeError('object attribute {0} is read-only'.format(attr))

        # Set the value of proper attribute
        else:
            # Transform all subdicts into Parameters dict
            if type(value) is dict:
                value = Parameters(_position = self._position + [attr], **value)

            # Put parameters in the right dict
            if attr[0] == '_' or attr in dir(self):
                object.__setattr__(self, attr, value)
            else:
                self.__dict__['params'][attr] = value

    def __repr__(self):
        return self.__dict__['params'].__repr__()

    def __str__(self):
        return self.__dict__['params'].__str__()

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, key):
        try:
            return self.__getattribute__(key)
        except AttributeError:
            return self.__getattr__(key)

    def __contains__(self, key):
        if key[0] == '_':
            return key in self.__dict__
        else:
            return key in self.__dict__['params']
        
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, s):
        self.__init__([i for i in s.items() if i[0] != 'params'], s['params'])

    def update(self, *args, **kwargs):
        '''
        Updates the parameters with the given list/object
        '''
        # Import dicts and lists and pairs
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

    def copy(self):
        return Parameters(dict(self).copy())

DEFAULTS = Parameters({
        'xlims': (0, None),
        'ylims': (0, None),
        'tlims': (0, None),
        'channel': 0,
        'max_processes': cpu_count(),
        'linkage': {
            'max_disp': (2, 2), # Maximum displacement, in pixels
            'max_blink': 25, # Maximum blinking, in frames
            'ambiguous_tracks': False, # Should we keep and try to solve ambiguous tracks?
            },
        'filtration': {
            'min_length': 3, # Minimum number of frames to keep a track
            'max_length': None, # The maximum number of frames to keep a track
            'mean_blink': 5, # The mean number of frames between two spots to keep a track
            },
        'barriers': {
            # Values for Charlie Brown DT: dbp = 12.5um and dbb = 37um
            'dbp': 47, # Distance between the barriers and the pedestals, in pixels
            'dbb': 139, # Distance between two sets of barriers, in pixels
            'axis': 'y', # The axis of the image with which the barriers are parallel. Either 'x' or 'y'
            'orientation': 'bp' # The order of appearance of barriers and pedestals, from the left. bp: barrier, then pedestals. pb: pedestals, then barriers
            },
        'barrier_detection': {
            'approx': 20, # Approximate 
            'blur': 2, # Blurring for diminishing the effect of noise.
            'guess': True, # Guess the position of barriers when there expected but not present
            'tlims': None # The frames to use for finding the barriers. If None, uses the main tlims
            },
        }, _use_defaults = False)

class Images(object):
    '''
    Wrapper for sources of images to support either bioformats or
    directory of images.

    Arguments:
    ---------
    - source: string. Path to the images. Can be:
        - A directory or a path (use wildcards compatible with Python's
        'glob' function to select specific files).
        - A file, the type must be compatible with the BioFormats
        library
    - channel: the input is the channel to use. Defaults to the first
    channel. The argument is only considered if 'source' is a file.
    - xlims: the region to select, in x. [first, last), in pixels.
    - ylims: the region to select, in y. [first, last), in pixels.
    - tlims: the frames. [first, last), in frames.
    The first image/pixel is indexed at 0.

    Methods:
    -------
    - read: generator function that will return images one by one
    - dimensions: size of the image in (x, y), in pixels
    - length: number of frames
    '''
    @property
    def xlims(self):
        return self.lims[0]
    @xlims.setter
    def xlims(self, lims):
        self.lims[0] = slice(*lims)
    @property
    def ylims(self):
        return self.lims[1]
    @ylims.setter
    def ylims(self, lims):
        self.lims[1] = slice(*lims)
    @property
    def tlims(self):
        return self.lims[2]
    @tlims.setter
    def tlims(self, lims):
        self.lims[2] = slice(lims[0], self.length(True)) if lims[1] is None else slice(*lims)

    def __init__(self, source, channel = 0, xlims = (0, None), ylims = (0, None), tlims = (0, None)):
        # A file was given
        if os.path.isfile(source) is True:
            self.is_file = True
            self.source = source
            self.channel = channel
            javabridge.start_vm(class_path=bioformats.JARS)
            self.reader = bioformats.ImageReader(self.source)

        # A path or dir was given
        else:
            if os.path.isdir(source) is True:
                wildcards = '*'
            else:
                source, wildcards = os.path.split(source)
                if os.path.isdir(source) is False:
                    raise ValueError('This path does not exists')
            self.is_file = False
            self.source = source + wildcards if source[-1] == '/' else source + '/' + wildcards

        # Set lims. Property setters will transform into proper slices
        self.lims = [None, None, None]
        self.xlims, self.ylims, self.tlims = xlims, ylims, tlims

    def read(self):
        '''
        Iterates over the images in the dir/file, returns a numpy array
        of each image, one by one.
        '''
        if self.is_file is False:
            for f in sorted(glob(self.source))[self.tlims]:
                yield io.imread(f)[self.ylims, self.xlims]
        else:
            for t in range(self.tlims.start, self.tlims.stop):
                yield self.reader.read(t = t, c = self.channel, rescale = False)[self.ylims, self.xlims]

    def get(self, frame):
        '''
        Returns a numpy array of the image at the given frame, or
        tlims[0]+frame if tlims is not False

        Arguments:
        ---------
        - frame: the frame to return, with 0 being the lower limit in
        the object's tlims (default=0)
        '''
        frame += self.tlims.start
        if self.is_file is False:
            return io.imread(sorted(glob(self.source))[frame])[self.ylims, self.xlims]
        else:
            return self.reader.read(t = frame, c = self.channel, rescale = False)[self.ylims, self.xlims]

    def dimensions(self, ignore_lims = False):
        '''
        Returns the shape of the images in the format (x, y), in pixels
        If ignore_lims is True: return the dimensions of the actual
        images, not of the ROI
        '''
        if self.is_file is False:
            shape = io.imread(glob(self.source)[0]).shape[::-1]
        else:
            shape = self.reader.rdr.getSizeX(), self.reader.rdr.getSizeY()

        if ignore_lims is True:
            return shape 
        else:
            if self.xlims.stop == None:
                x = shape[0] - self.xlims.start
            else:
                x = min(self.xlims.stop - self.xlims.start, shape[0] - self.xlims.start)
            if self.ylims.stop == None:
                y = shape[1] - self.ylims.start
            else:
                y = min(self.ylims.stop - self.ylims.start, shape[1] - self.ylims.start)
            return x, y

    def length(self, ignore_lims = False):
        '''
        Return the number of frames in the experiment
        If ignore_lims is True: return the length of the actual set,
        not of the ROI
        '''
        if self.is_file is False:
            l = len(glob(self.source))
        else:
            l = self.reader.rdr.getSizeT()

        if ignore_lims is True:
            return l 
        else:
            return min(self.tlims.stop - self.tlims.start, l - self.tlims.start)

class Dataset(Parameters):
    '''
    A dataset is a collection of data that have the same experimental
    conditions, including:
        - x, y, t offset
        - source
        - channel
        - experimental settings

    Datasets can take unlimited, user-defined parameters, including:
        - source: path, including wildcards for selecting specific files
        or BioFormat compatible file containing the images
        - xlims: the region of interest' x location. [first, last), in
        pixels. Default: all
        - ylims: the region of interest' y location. [first, last), in
        pixels. Default: all
        - tlims: the frames of interest. [first, last), in frames.
        Default: all
        - channel: the channel of the image. Default: 0
        - max_processes: the maximum number of processes to spawn when
        multiprocessing. Default: number of CPUs.
    For xlims, ylims and tlims, the first image/pixel is indexed at 0.
    Set the outer boundary to 'None' to end limit at the last
    image/pixel.

    To set or modify a parameter, simply call it as an attribute or as
    an item of the dataset, or during instantiation. Examples:
        D = Dataset('file.nd2', comment = 'This works, but only if the
        source is the first argument')
        D = Dataset(source = 'file.tif')
        D = Dataset({source: 'file.tif'}, channel = 0)
        D = Dataset(('source', 'file.tif'), channel = 0)
        D.source = 'dir_of_data/*.tif'
        D['source'] = '*c1*.jpg'
            ...
    Any pair of (parameter, value), or list of pairs, or dictionary,
    or object with a getitems() function, can be passed during
    instantiation.
    '''

    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__()
        self.images = None
        self.spots = None
        self.tracks = None
        # If the first argument is a string, it is the source
        if len(args) > 0 and type(args[0]) is str:
            args = list(args)
            self.source = args.pop(0)
        self.update(*args, **kwargs)

    def __getstate__(self):
        d = dict([i for i in self.__dict__.iteritems() if i[0] != 'params'])
        d['params'] = dict([i for i in self.__dict__['params'].iteritems() if i[0] != 'images'])
        return d

    @property
    def source(self):
        return self.__dict__['params']['source']
    @source.setter
    def source(self, f):
        self.images = Images(f, self.channel, self.xlims, self.ylims, self.tlims)
        self.__dict__['params']['source'] = f

    @property
    def xlims(self):
        return self.__dict__['params']['xlims'] if 'xlims' in self else DEFAULTS['xlims']
    @xlims.setter
    def xlims(self, v):
        if self.images is not None:
            setattr(self.images, 'xlims', v)
        self.__dict__['params']['xlims'] = v
    @property
    def ylims(self):
        return self.__dict__['params']['ylims'] if 'ylims' in self else DEFAULTS['ylims']
    @ylims.setter
    def ylims(self, v):
        if self.images is not None:
            setattr(self.images, 'ylims', v)
        self.__dict__['params']['ylims'] = v
    @property
    def tlims(self):
        return self.__dict__['params']['tlims'] if 'tlims' in self else DEFAULTS['tlims']
    @tlims.setter
    def tlims(self, v):
        if self.images is not None:
            setattr(self.images, 'tlims', v)
        self.__dict__['params']['tlims'] = v

    def save(self, f = None):
        '''
        Saves the dataset, in its current form.
        Saves:  the parameters
                the spots
                the tracks
        A Dataset saved with save() and loaded with load() should be
        in the same state as left the last time it was saved.

        Arguments:
        ---------
        - f: file to save the dataset to. If None, the function returns
        the content of the would-be file
        '''
        content = pickle.dumps(self)
        if f is not None:
            with open(f, 'w') as data:
                data.write(content)
            self.file = f
        return content if f is None else None

    def load(self, f):
        '''
        Loads a Dataset from a file or a string.
        A Dataset saved with save() and loaded with load() should be
        in the same state as left the last time it was saved.

        Arguments:
        ---------
        - f: file or string to load the dataset from.
        '''
        if os.path.isfile(f) is True:
            with open(f, 'r') as data:
                D = pickle.load(data)
            self.file = f
        else:
            D = pickle.loads(f)
        self.update([i for i in D.__dict__.iteritems() if i[0] != 'params'], D)

    def test_detection_conditions(self, blur, threshold, frame, output = None, save = True):
        '''
        Tests conditions for spots detection.
        Returns a RGB image with a red pixel on each detected Gaussian center.

        Arguments:
        ---------
        - blur: the standard deviation of the Gaussian kernel for
        blurring (understand "cleaning up") the image. Should be around
        the size of a spot (in pixels) divided by 2.
        - threshold: the threshold to detect spots in the
        skimage.feature.blob_log function
        - frame: the frame to use, from the images, to test the conditions
        - output: output file. If None, opens a window with the marked image
        - save: whether to save (True) or not (False) the given
        detection conditions as Dataset's defaults.
        '''
        image = self.images.get(frame)

        # Find spots
        t = time()
        spots = find_blobs(image, blur, threshold)
        print('Found {0} spots in {1:.2f}s\r'.format(len(spots), time()-t))

        # Prepare the output image
        shape = list(image.shape)
        ni = np.zeros([3] + shape, dtype=np.uint8)
        ni[...,:] = image/image.max()*255
        ni = ni.transpose(1, 2, 0)

        # Mark spots
        for s in spots:
            ni[s[1],s[0],:] = (255, 0, 0)

        # Show/save image
        if output is None:
            io.imshow(ni, cmap=plt.cm.gray)
            io.show()
        else:
            io.imsave(output, ni)

        if save is True:
            if 'detection' in self:
                self.detection.update({'blur': blur, 'threshold': threshold})
            else:
                self.detection = {'blur': blur, 'threshold': threshold}
        return spots

    def detect_spots(self, verbose = True):
        '''
        Finds the blobs on the Dataset's images. 
        Uses values from Dataset.detection, that can be set using
        test_detection_conditions()
        The number of simultaneous processes can be modified using
        Dataset.max_processes, which defaults to the number of CPUs
        (max speed, at the cost of slowing down all other work)

        Arguments:
        ---------
        - verbose: Writes about the time and frames and stuff as it works
        '''

        # Multiprocess through it
        t = time()
        spots, pool = list(), Pool(self.max_processes)
        for blobs in pool.imap(find_blobs, iter((i, self.detection.blur, self.detection.threshold, j) for j, i in enumerate(self.images.read()))):
            if verbose is True:
                print('\rFound {0} spots in frame {1}. Process started {2:.2f}s ago.         '.format(len(blobs), blobs[0][4], time() - t), end='')
            stdout.flush()
            spots.extend(blobs)
        pool.close()
        pool.join()

        # Save the spot list into memory
        if verbose is True:
            print('\nFound {0} spots in {1} frames in {2:.2f}s'.format(len(spots), self.images.length(), time() - t))
        self.spots = np.array(spots, dtype = {'names': ('x', 'y', 's', 'i', 't'), 'formats': (float, float, float, int, int)}).view(np.recarray)
        return spots

    def link_spots(self, verbose = True):
        '''
        Link spots together in time

        Arguments:
        ---------
        - verbose: Writes about the time and frames and stuff as it works
        '''

        # Reorganize spots by frame and prepare the Graph
        G = nx.DiGraph()
        n_frames = self.images.length()
        frames = [[] for f in range(n_frames)]
        for i, spot in enumerate(self.spots):
            frames[spot['t']].append((spot['x'], spot['y'], i))
            G.add_node(i, frame=spot['t'])

        # Make optimal pairs for all acceptable frame intervals (as defined in max_blink)
        for delta in range(1, self.linkage.max_blink+1):

            if verbose is True: 
                print('\rDelta frames: {0}'.format(delta), end='')
                stdout.flush()
            for f in range(n_frames - delta):
                # Measure the distances between spots
                d = np.abs(np.array(frames[f])[:, np.newaxis,:2] - np.array(frames[f+delta])[:,:2])

                # Filter out the spots with distances that excess max_disp in x and/or y
                disp_filter = d - self.linkage.max_disp >= 0
                disp_mask = np.logical_or(disp_filter[:,:,0], disp_filter[:,:,1])

                # Reduce the 3rd dimension
                d = np.sqrt(d[:,:,0]**2+d[:,:,1]**2)

                # Find the optimal pairs
                f_best = d != np.min(d, axis=0)
                fd_best = (d.T != np.min(d, axis=1)).T
                mask = np.logical_or(f_best, fd_best)
                mask = np.logical_or(mask, disp_mask)
                pairs = ma.array(d, mask=mask)

                # Organize in pairs, or edges, for graph purposes
                for s1, s2 in np.array(np.where(pairs.mask == False)).T:
                    G.add_edge(frames[f][s1][2], frames[f+delta][s2][2], weight=d[s1][s2])

        # Only keep the tracks that are not ambiguous (1 spot per frame max)
        tracks, a_tracks = list(), list()
        for track in nx.weakly_connected_component_subgraphs(G):
            t_frames = np.array(sorted([(s, self.spots[s]['t']) for s in track], key=lambda a:a[1]))

            # Good tracks
            if len(t_frames[:, 1]) == len(set(t_frames[:, 1])):
                tracks.append(sorted(track.nodes(), key=lambda s: self.spots[s]['t']))

            # Ambiguous tracks
            elif self.linkage.ambiguous_tracks is True:
                print(track.number_of_nodes(), track.number_of_edges())
                track = transitive_reduction(track, t_frames[:, 0])
                sources, junctions = list(), list()
                for i in track.in_degree().items():
                    if i[1] == 0:
                        sources.append(i[0])
                    elif i[1] > 1:
                        junctions.append(i[0])
                sinks = [i[0] for i in track.out_degree().items() if i[1] == 0]
                a_tracks.append(sorted(track.nodes(), key=lambda s: spots[s][3]))

                print(track.number_of_nodes(), track.number_of_edges())
                print(sources)
                print(junctions)
                print(sinks)
                a = np.array([(s, spots[s][3], spots[s][0], spots[s][1]) for s in sorted(track.nodes(), key=lambda a:spots[a][3])])
                print([i for i in nx.weakly_connected_components(track)])
                for b, c, d, e in a:
                    print(b, '\t', c, '\t', d, '\t', e)
                sl, f, x, y = a.T
                plt.figure()
                plt.subplot(211)
                plt.scatter(x, y, c=f)
                plt.subplot(212)
                nx.draw_networkx(track)
                plt.show()

        if verbose is True: print('\nFound {0} tracks'.format(len(tracks)))

        if self.linkage.ambiguous_tracks == False:
            self.tracks = tracks
            return tracks
        else:
            return tracks, a_tracks

    def filter(self, overwrite = True, parameters = None):
        '''
        Filters the tracks obtained by link_spots based on parameters
        set either in self.filtration, either passed as
        arguments.

        Arguments:
        ---------
        - overwrite: bool. If True, the Dataset's tracks will be
        replaced by the filtered ones.
        - parameters: dict. Parameters to use for filtering. If None,
        will use the ones from the Dataset's parameters
        (Dataset.filtration).
        '''

        if parameters is None: parameters = self.filtration
        if parameters['max_length'] is None: parameters['max_length'] = self.images.length()
        n_tracks = list()
        for track in self.tracks:
            if  parameters['min_length'] <= len(track) <= parameters['max_length']:
                if np.abs(np.diff(sorted([self.spots[s]['t'] for s in track]))).mean() <= parameters['mean_blink']:
                    n_tracks.append(track)

        if overwrite is True:
            self.filtration = parameters
            self.tracks = n_tracks

        return n_tracks

    def draw(self): # NOT IMPLEMENTED YET
        '''
        Draws the tracks and/or spots onto the images of the dataset.
        '''

    def get_tracks(self, properties = None):
        '''
        Returns the tracks in the Dataset as a list of spots, rather
        than a list of spot ids.

        Arguments:
        ---------
        - properties: list of the properties to return for each spots.
        Will return the properties in the same order as given. If None,
        returns all properties.
            - x: the position, in x
            - y: the position, in y
            - s: the standard deviation of the gaussian kernel
            - i: the intensity at (x, y)
            - t: the frame
            Example list: ['x', 'y', 't']
        '''
        spots = self.spots[properties] if properties is not None else self.spots
        tracks = list()
        for track in self.tracks:
            t = list()
            for s in track:
                t.append(spots[s])
            tracks.append(t)
        return tracks

    def export_track(self, track, properties = None):
        '''
        Return the given track as a list of spot properties rather than
        as a list of spot ids.

        Arguments:
        - track: a list of spot ids from the dataset
        - properties: list of the properties to return for each spots.
        Will return the properties in the same order as given. If None,
        returns all properties.
            - x: the position, in x
            - y: the position, in y
            - s: the standard deviation of the gaussian kernel
            - i: the intensity at (x, y)
            - t: the frame
        '''
        spots = self.spots[properties] if properties is not None else self.spots
        t = list()
        for s in track:
            t.append(spots[s])
        return t

    def as_isbi_xml(self, f = 'result.xml', snr = 7, density = 'low', scenario = 'VIRUS'):
        root = ET.Element('root')
        t = ET.SubElement(root, 'TrackContestISBI2012')
        t.attrib = {'SNR': str(snr), 'density': density, 'scenario': scenario}
        for track in self.get_tracks():
            p = ET.SubElement(t, 'particle')
            for spot in track:
                s = ET.SubElement(p, 'detection')
                s.attrib = {'t': str(spot['t']), 'x': str(spot['x']), 'y': str(spot['y']),  'z': '0'}
        E = ET.ElementTree(element = root)
        E.write(f)

def find_blobs(*args):
    '''
    Finds blobs in an image. Returns a list of spots as (y, x, s, i),
    where y and x are the locations of the center, s is the standard
    deviation of the gaussian kernel and i is the intensity at the
    center.

    Arguments:
    ---------
    args: (optionally, for multiprocessing, list of:)
    - image: a numpy array representing the image to analyze
    - blur: see ndimage.gaussian_filter()'s 'sigma' argument
    - threshold: see scipy.feature.blob_log()'s 'threshold' argument
    - extra: information to be added at the end of the blob's properties
    '''
    args = args[0] if len(args) == 1 else args
    image, blur, threshold, extra = args[0], args[1], args[2], args[3:]
    b = blob_log(ndimage.gaussian_filter(image, blur), threshold=threshold, min_sigma=1)
    blobs = list()
    for y, x, s in b:
        blob = [x, y, s, image[y][x]]
        blob.extend(extra)
        blobs.append(tuple(blob))
    return blobs

class Experiment(object):
    '''
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
    '''

    def __init__(self, *args, **kwargs):
        self.datasets = []
        self.parameters = Parameters()

        # Only one argument given: path to the dataset
        if len(args) == 1:
            try:
                load = tarfile.is_tarfile(args[0])
            except:
                load = False
            if load == True:
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
        '''
        Special handling for parameters
        The Experiment class passes down all given parameters to its
        Datasets, for the best and for the worst. With great powers
        come great responsibilities.
        '''
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
            raise AttributeError("Cannot set a source on an Experimentthat is already initialized. Use Experiment.add_dataset() instead.")

    def add_dataset(self, *args, **kwargs):
        '''
        Adds a dataset to the experiment

        Possible arguments:
        ------------------
        A) Dataset object. The dataset will directly be added to the Experiment. 
        B) list/dict of parameters or keywords: the Dataset will receive
        the Experiment's parameters first, and then those provided
        C) string (Path to the source of images) + list/keywords
        parameters. The Dataset source will be set with the string (has
        to be the first argument). Keywords or lists/dicts can be added
        as parameters, that will be added on top of the Experiment's
        parameters.
        '''
        args = list(args)
        if len(args) == 1 and isinstance(args[0], Dataset):
            self.datasets.append(args[0])
        else:
            d = Dataset(self.parameters)
            if len(args) > 0 and type(args[0]) == str:
                d.source = args.pop()
            d.update(*args, **kwargs)
            self.datasets.append(d)

    def save(self, f = None):
        '''
        Save the Experiment, its datasets and parameters.
        An Experiment saved with save() and loaded with load() should
        be in the same state as left the last time it was saved.

        The saved file will actually be an archive of the pickled objects.

        Arguments:
        ---------
        - f: file to save the experiment to. If None, the experiment
        is saved to the last file it was saved to or loaded from, which
        is written in the 'file' parameter of the experiment. If this
        parameter does not exist, the function returns the content of
        the would-be file as an array of pickled objects.
        '''
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
                    i = tarfile.TarInfo(name = name)
                    i.size = len(c.buf)
                    t.addfile(tarinfo = i, fileobj = c)
            self.file = os.path.abspath(f)
        return content if f is None else None

    def load(self, f = None):
        '''
        Loads an Experiment from a file or a string. An Experiment saved
        with save() and loaded with load() should be in the same state
        as left the last time it was saved.
        
        Arguments: 
        --------- 
        - f: file or array of pickled objects to load the experiment
        from. If None, the experiment is loaded from the last file it
        was saved to or loaded from, which is written in the 'file'
        parameter of the experiment.
        '''
        if f is None:
            if 'file' in self:
                f = self.file
            else:
                raise AttributeError('The experiment has no file associated and no input parameter was given.')
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

    # Spots and Tracks
    def test_detection_conditions(self): # NOT IMPLEMENTED YET
        '''
        Finds the optimal detection conditions in each of the underlying
        dataset.
        '''

    def find_spots(self, verbose = True):
        '''
        Detects spots in each of the underlying datasets. See
        Dataset.detect_spots
        '''
        t, n = time(), 0
        for i, ds in enumerate(self.datasets):
            if verbose is True: print('Working on dataset {0}:'.format(i))
            ds.detect_spots(verbose)
            n += len(ds.spots)
        if verbose is True: print('Found {0} spots in {1} datasets in {2:.2f}s:\n'.format(n, i + 1, time() - t))
        return True

    def link_spots(self, verbose = True):
        '''
        Links spots to form tracks in each of the underlying datasets. 
        See Dataset.link_spots
        '''
        t, n = time(), 0
        for i, ds in enumerate(self.datasets):
            if verbose is True: 
                print('Working on dataset {0}:'.format(i))
                stdout.flush()
            ds.link_spots(verbose)
            n += len(ds.tracks)
        if verbose is True: 
            print('\rFound {0} tracks in {1} datasets in {2:.2f}s:\n'.format(n, i + 1, time() - t), end = '')
            stdout.flush()
        return True

    def filter(self, overwrite = True, parameters = None):
        '''
        Filter tracks in each of the underlying datasets. 
        See Dataset.link_spots
        '''
        filtered_tracks = list()
        for ds in self.datasets:
            tracks = ds.filter(overwrite, parameters)
            for track in tracks:
                filtered_tracks.append(ds.export_track(track))
        return filtered_tracks

    def get_tracks(self, properties = None):
        '''
        Returns the tracks from each dataset of the experiment as a
        list of spot properties

        Arguments:
        ---------
        - properties: list of the properties to return for each spots.
        Will return the properties in the same order as given. If None,
        returns all properties.
            - x: the position, in x
            - y: the position, in y
            - s: the standard deviation of the gaussian kernel
            - i: the intensity at (x, y)
            - t: the frame
            Example list: ['x', 'y', 't']
        '''
        tracks = list()
        for ds in self.datasets:
            tracks.extend(ds.get_tracks(properties))
        return tracks

    # Transformation of images into datasets
    def find_barriers(self, datasets = None, overwrite = True):
        '''
        Finds the barriers in an experiment.

        NEED TO IMPLEMENT:
            - IMCOMPLETE BARRIER SETS

        Arguments:
        --------
        - datasets: list of the ids of the datasets to find barriers
        in. If None, tries all datasets
        - overwrite: bool. If True, the given datasets will be replaced
        by one dataset for each set of their barriers.
        '''
        barrier_datasets = list()
        if datasets is None: datasets = range(len(self.datasets))
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
            for i in range(0, len(d_proj)-1):
                neighbors = sorted((d_proj[i], d_proj[i+1]))
                if neighbors[0] < 0 and neighbors[1] > 0:
                    tangents.extend([i])
            extremas = np.where(np.square(np.diff(d_proj)) - np.square(d_proj[:-1]) >= 0)[0]
            peaks = sorted([p for p in tangents if p in extremas])
            distances = np.subtract.outer(peaks, peaks)

            # Build sets of barriers
            max_sets = int(ceil(len(projection) - dbp)/dbb) + 1
            exp_dists, sets = [dbp], list()
            for i in range(1, max_sets):
                exp_dists.extend([i*dbb - dbp, i*dbb])

            #Find the best possible match for consecutive barriers-pedestals for each peak
            for peak in range(distances.shape[0]):
                i = 0 # Position in the sets of barriers
                j, last_j = 0, -1 # Distance with last peak
                barriers_set, current_set = list(), list()
                a = 0 # Allowed difference for ideal distance
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
                score = sum([i[1] for i in s])
                set_scores.append((len(s), -score))
            self.barriers.positions = sorted([sorted(i[0]) for i in sets[set_scores.index(sorted(set_scores, reverse=True)[0])]])

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

    def correct_rotation(source, destination, bins=10): # NOT TRANSFERED TO OOP
        '''
        Corrects rotation on images based on the barriers
        Assumes that rotation is constant over time.

        source: the directory containing the images
        destination: where to write the corrected images
        bins: number of bins to split the image in, vertically (lower if noise is higher)
        '''

        images = sorted(os.listdir(source))
        shape = io.imread(source+'/'+images[0]).shape
        bins = np.linspace(0, shape[1], bins, dtype=int)
        barriers = list()
        for i in range(len(bins) - 1):
            #frames = (0, len(images)), 
            barriers.append([i for j in find_barriers(source, ylims=(bins[i], bins[i+1])) for i in j])

        m = list()
        for y in np.array(barriers).T:
            y -= y[0]
            m.append(np.polyfit(bins[1:], y, 1)[0])
        angle = degrees(atan(np.mean(m)))

        for i in images:
            io.imsave(destination+'/'+i, rotate(io.imread(source+'/'+i), angle, mode='nearest', preserve_range=True).astype(dtype=np.int16))

        return True

    # Analysis
    def sy_plot(self, binsize = 3):
        '''
        Draws a SY (Stacked Ys) plot based on the tracks
        '''

        # Define the limits in X
        lims = (0, max([s for t in self.get_tracks('x') for s in t]))

        # Create a dic of position -> lengths of tracks
        bins = np.arange(0, lims[1], binsize)
        y = dict()
        for i in range(0, len(bins)):
            y[i] = list()

        n = 0
        for track in self.get_tracks(['x', 't']):
            frames = [spot['t'] for spot in track]
            length = max(frames) - min(frames) + 1
            position = np.mean([spot['x'] for spot in track])
            b = max([i for i, b in enumerate(bins) if position - b >= 0])
            y[b].append(np.log(length))
            n += 1

        # Build a masked array to feed plt.pcolor with
        max_length = max([len(t) for t in y.values()])
        C = np.ma.zeros((len(bins), max_length))
        mask = np.ones((len(bins), max_length), dtype=np.bool)
        for x, data in y.items():
            datapoints = len(y[x])
            # Unmask 
            mask[x, 0:datapoints] = False
            # Fill the column
            C[x, 0:datapoints] = sorted(y[x])

        C.mask = mask
        plt.figure()
        plt.pcolor(C.T)
        cb = plt.colorbar()
        cb.set_label('Length of the interaction (frames)')
        lengths = [v for l in y.values() for v in l]
        cb_ticks = np.linspace(min(lengths), max(lengths), 10)
        cb.set_ticks(cb_ticks)
        cb.set_ticklabels([int(round(np.e**v, 0)) for v in cb_ticks])
        ticks = range(0, len(bins), 2)
        plt.xticks(ticks, [int(bins[i]) for i in ticks])
        plt.figtext(0.25, 0.87, ha='right', va='top', s='N={0}'.format(n))
        plt.xlabel('Position on DNA (pixel)')
        plt.ylabel('Number of interactions')
        plt.savefig('sy_plot.png')
        plt.close()

        return bins, y

    def histogram(self, prop = 'x', binsize = 3):
        '''
        Survival plot
        '''
        if prop == 'l':
            tracks = self.get_tracks('t')
            data = [max(track) - min(track) for track in tracks]
        else:
            tracks = self.get_tracks(prop)
            data = [np.mean(track) for track in tracks]
        bins = int(ceil((max(data)  - min(data))/binsize))
        plt.hist(data, bins=bins)
        plt.show()


# Refine the spots to subpixel resolution
def subpixel_resolution(spots, images_source):
    # Reorganize spots by frame
    n_frames = np.array(spots, dtype='i8')[:, 3].max() + 1
    frames = [[] for f in range(n_frames)]
    i = 0
    for spot in spots:
        frames[spot[3]].append(list(spot)+[i])
        i += 1

    # Fit a 2D gaussian on every single spot
    i = 0
    #r = np.sqrt(2)
    r=4
    spr = [[] for s in spots]
    for f in sorted(os.listdir(images_source)):
        image = io.imread(d+'/'+f)
        xlims = (0, image.shape[1])
        ylims = (0, image.shape[0])
        for s in frames[i]:
            y = [max(int(floor(s[1] - r*s[2])), ylims[0]), 
                 min(int(ceil(s[1] + r*s[2]))+1, ylims[1])]
            x = [max(int(floor(s[0] - r*s[2])), xlims[0]), 
                 min(int(ceil(s[0] + r*s[2]))+1, xlims[1])]
            data = image[y[0]:y[1],x[0]:x[1]]
            coords = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
            try:
                fit, cov = curve_fit(gaussian_2d, coords, data.ravel(), p0=(image[s[1]][s[0]], s[0]-x[0], s[1]-y[0], s[2]))
                print(s[:2], [fit[1]+x[0], y[0]+fit[2]], np.sqrt(np.diag(cov)))
                plt.imshow(data, cmap=plt.cm.gray, origin='bottom')
                plt.plot(fit[1], fit[2], 'ro')
                plt.plot(s[0]-x[0], s[1]-y[0], 'go')
                plt.show()
                s = [x[0]+fit[1], y[0]+fit[2], fit[3], i, s[4]]
            except:
                pass
            spr[s[4]] = s
            #print(spr[s[4]])
        i += 1

    return spr

def gaussian_2d(coords, A, x0, y0, s):
    x, y = coords
    return (A*np.exp(((x-x0)**2+(y-y0)**2)/(-2*s**2))).ravel()

# Draw tracks
def draw_tracks(tracks, spots, d, destination, draw_spots = False):
    n_frames = np.array(spots, dtype=int)[:, 3].max() + 1

    # Make colors
    colors = [np.array(hls_to_rgb(randrange(0, 360)/360, randrange(20, 80, 1)/100, randrange(20, 80, 10)/100))*255 for i in tracks]

    # Separate tracks per image, because memory is short
    frames = [[] for f in range(n_frames)]
    i = 0
    for track in tracks:
        for s in track:
            s = spots[s]
            spot = list(s[:2])
            spot.extend(colors[i])
            frames[s[3]].append(spot)
        i += 1

    # Draw all spots
    if draw_spots is True:
        used = set([spot for track in tracks for spot in track])
        unused = used.symmetric_difference(range(0, len(spots)))
        for s in unused:
            s = spots[s]
            spot = list(s[:2])
            spot.extend([255, 0, 0])
            frames[s[3]].append(spot)

    # Write tracks RGB images
    j = 0
    for f in sorted(os.listdir(d)):

        # Make images RGB
        i = io.imread(d+'/'+f)
        shape = list(i.shape)
        ni = np.zeros([3] + shape)
        ni[...,:] = i/i.max()*255
        ni = ni.transpose(1, 2, 0)

        # Write spots
        for s in frames[j]:
            ni[s[1],s[0],:] = s[2:]
        misc.imsave(os.path.normpath(destination)+'/'+str(j)+'.tif', ni)
        j += 1


# Draws a histogram of position on DNA
def histogram(tracks, spots, binsize=3, prop=0, use_tracks = True):
    lims = (0, max([spot[prop] for spot in spots]))
    bins = int(ceil((lims[1]  - lims[0])/binsize))

    if use_tracks is False:
        data = [spot[prop] for spot in spots]
    else:
        data = [np.mean([spots[spot][prop] for spot in track]) for track in tracks]

    plt.figure()
    plt.xlabel('Property {0} of the spot'.format(prop))
    plt.ylabel('Number of interactions')
    plt.hist(data, bins=bins)
    plt.savefig('histogram.png')
    plt.close()

# Zhi plot drawing function: dwell time vs initial binding time
def zhi_plot(tracks, spots):
    fps = 1/0.06
    binsize = int(ceil(1000/fps))
    exp_len = int(ceil(10000/fps))

    bins = [[] for t in range(0, exp_len, binsize)]
    for track in tracks:
        frames = [spots[s][3] for s in track]
        m = min(frames)
        dt = (max(frames) - m)/fps
        bins[int(floor((m/fps)/binsize))].append(dt)

    # All the dwell times
    xs = list()
    ys = list()
    for i, b in enumerate(bins):
        xs.extend([i*binsize for d in b])
        ys.extend(b)
    plt.plot(xs, ys, 'g.', label='Calculated dwell times')

    # Mean dwell tims
    y = [np.mean(b) for b in bins]
    x = range(0, exp_len, binsize)
    plt.plot(x, y, 'ro', label='Mean dwell time: {0:.2f}s'.format(np.mean(ys)))
    plt.ylabel('Mean dwell time (s)')
    plt.yscale('log')
    plt.xlabel('Binned time of initial binding (s)')
    plt.legend()
    plt.figtext(0.82, 0.12, 'N={0}'.format(len(tracks)))
    plt.savefig('150720. Dwell time vs initial binding time.png')

# Return the transitive reduction of a given graph
def transitive_reduction(G, order = None, adj_matrix = False):
    '''
    Returns the transitive reduction of a given graph
    Based on Aho, A. V., Garey, M. R., Ullman, J. D. The Transitive
    Reduction of a Directed Graph. SIAM Journal on Computing 1, 131–137
    (1972).

    Arguments:
    ----------
    - order: the order in which the vertices appear in time (to fit the
    second condition). If None, uses the same order as in the graph
    - adj_matrix: returns an adjacency matrix if True, a Graph if False
    '''

    # Transitive closure
    MT = transitive_closure(G, order, True)

    # Reorganize the adjacency matrix
    V = G.nodes()
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

# Returns the transitive closure of a given Graph
def transitive_closure(G, order = None, adj_matrix = False):
    '''
    This method assumes that your graph is:
        1. Directed acyclic
        2. Organized so that a node can only interact with a node
        positioned after it, in the adjacency matrix

    Arguments:
    ----------
    - order: the order in which the vertices appear in time (to fit the
    second condition)
    - adj_matrix: returns an adjacency matrix if True, a Graph if False
    '''
    V = G.nodes()
    M = nx.to_numpy_matrix(G, nodelist=order, weight=None)

    # Close the graph
    for i, j in sorted((np.array(np.where(M==1), dtype=int).T)[0], key=lambda a:[0])[::-1]:
        M[i] = np.logical_or(M[i], M[j])

    # Return in the proper format
    if adj_matrix is True:
        return M
    else:
        GT = G.copy()
        for i, j in (np.array(np.where(M == 1)).T):
            GT.add_edge(order[i], order[j])
        return GT

def import_isbi_data(xml):
    tracks, spots, i = list(), list(), 0
    X = ET.parse(xml)
    R = X.getroot()
    for particle in R[0]:
        t = list()
        for spot in particle:
            spots.append((spot.attrib['x'], spot.attrib['y'], spot.attrib['t']))
            t.append(i)
            i += 1
        tracks.append(t)
    D = Dataset(source = os.path.split(xml)[0])
    D.spots =  np.array(spots, dtype = {'names': ('x', 'y', 't'), 'formats': (float, float, int)}).view(np.recarray)
    D.tracks = tracks
    
    return D

# Output
if __name__ == '__main__':

    data1 = '/home/corentin/H2A ATTO532 data/'
    data2 = '/home/corentin/ISBI Challenge/VESICLE/VESICLE snr 7 density mid/*.tif'
    f = '/home/corentin/Dropbox/Code/CATS/H2A ATTO532 data.exp'

    E = Experiment(data2)
    # E.datasets[0].test_detection_conditions(0.5, 0.03, 10, 'test.tif')
    # E.detection = {'blur': 0.5, 'threshold': 0.03}
    # E.find_spots()
    E.load('/home/corentin/Dropbox/Code/CATS/vesicles snr 7 mid.exp')
    E.filtration = DEFAULTS.filtration.copy()
    E.filtration.min_length = 4
    E.filter()
    E.datasets[0].as_isbi_xml()
    
    # T = import_isbi_data('/home/corentin/ISBI Challenge/VESICLE/VESICLE snr 7 density mid/VESICLE snr 7 density mid.xml')
    # E.find_barriers()
    # E.detection = {'blur': 0.5, 'threshold': 0.007}
    # E.find_spots()
    # E.save(f)
    # E.tlims = (0, 225)
    # E.detection = {'blur': 0.5, 'threshold': 0.005}
    # E.find_barriers()
    # print([e.xlims for e in E.datasets])
    # E.find_spots()
    # E.save('H2A ATTO532 data.exp')
    
    #D = Dataset()
    #D.source = data
    #D.channel = 1
    #D.max_processes = 1
    #D.xlims = (12, 24)
    #D.tlims = (0, 10)
    #D.test_detection_conditions(0.5, 0.007, 1)
    #D.detection = {'blur': 0.5, 'threshold': 0.01}
    #D.detect_spots()
    #D.link_spots()
    #D.save('H2A ATTO532 data.ds')

    #E.datasets[0].load('H2A ATTO532 data.ds')
    #E.find_barriers()
    #E.detect_spots()
    #E.link_spots()
    #E.filter()
    #E.histogram('l', binsize = 10)

    # Stuff to use: nucs
    #spots_f = '/home/corentin/H2AATTO532.spots'
    #tracks_f = '/home/corentin/H2AATTO532.tracks'
    #d = '/home/corentin/H2A ATTO532 data/'
    #destination = '/home/corentin/H2A ATTO532 data results/'


    # javabridge.kill_vm()