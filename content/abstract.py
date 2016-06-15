# -*- coding: utf8 -*-
"""
Basic classes, meant to be subclassed, to deal with content.

contents represents a list of individual units of content
content reprensents one of these units and inherit from NumPy's recarrays to organize detections.
Detection represent a subunit of content, found in one single frame for example.

Hence, Detections aggregate into contents which aggregate into contents.
"""
from __future__ import absolute_import, division, print_function
from .. import extensions, defaults, sources
from ..adict import dadict
import numpy as np
from copy import deepcopy
import pickle
import os

__all__ = ['contents', 'content', 'detection']


@extensions.append
class contents(list, dadict):

    """
    More or less abstract class for any type of content.

    contents classes inherit from lists and dadict. They contain dictionary items and list items. The __iter__ function (used in iterations, like  for loops) returns the content of the list (i.e.: the actual content), whereas the content of the dict (parameters, attributes, etc.) can be accessed via the usual dict methods items(), keys(), values() and their iterative equivalents.

    Example:
        c = contents(source1, source2, processor=processors.common, option1='value')
        func(*c)    # Unpacks the list
        func(**c)   # Unpacks the dict
    """

    def __init__(self, *args, **kwargs):
        """
        Initiate the object.

        During initialization of the object, the keyword arguments are first set, except 'sources', and then the sources are set from the non-keyword arguments and the keyword argument 'sources'.

        Arguments:
            Paths to the sources or source object (cats.sources.Images or cats.sources.ROI). Paths will be transformed into cats.sources.Images objects.

        Keyword arguments:
            - sources: list of sources (cats.sources.ROI, cats.sources.Images or paths to the source.) If only one source, it can be given as is, without puting it in a list.
            - processor: Mandatory, unless set it the defaults (contents objects inherit from cats.adict.dadict) The processor (see doc from the cats.processors module) to use to obtain the particles.
            - abs_position: (bool) wheither the positions returned by the content should be absolute to the images limits or to the ROI limits, if applicable.
            - units: (dict) the units to return the content values in, as 'column': 'units' (example: 'x': 'um')
            - any argument for the processor to use
            - whatever you want, really.
        """
        dadict.__init__(self)

        # Set the _defaults dict
        if self.__class__.__name__ in dir(defaults):
            self._defaults = getattr(defaults, self.__class__.__name__)

        # Set arguments
        for k, v in kwargs.items():
            if k != 'sources':
                setattr(self, k, v)

        # Set sources
        if 'sources' in kwargs:
            sources = kwargs['sources']
        elif len(args) != 0:
            sources = args
        else:
            sources = []
        self.sources = sources

        # Default to relative positions if need be
        if 'abs_position' not in kwargs:
            self.abs_position = False

    def __repr__(self):
        """Mix representations from list and dadict."""
        return list.__repr__(self) + '\n' + dadict.__repr__(self)

    def __getitem__(self, item):
        """Orchestrate items between list and dict."""
        if type(item) is not str:
            return list.__getitem__(self, item)
        else:
            return dadict.__getitem__(self, item)

    def copy(self):
        """Copy the list and dict parts of the object."""
        r = type(self)(self.__dict__.copy())
        r.extend([i for i in self])
        return r

    def deepcopy(self):
        """Copy the list and dict parts of the object without leaving references."""
        return deepcopy(self)

    @property
    def sources(self):
        """
        List of sources.

        The value can be set with a list or single instance of:
            paths to images (interpretable by cats.sources.Images),
            cats.sources.Images
            cats.sources.ROI
        All other input will lead to errors.
        A single instance will be transformed into a list.

        """
        return self.__getprop__('sources')

    @sources.setter
    def sources(self, value):
        """Add/Set sources."""
        # A. Transform input into proper sources
        if type(value) is str or isinstance(value, sources.Images):
            value = [value]
        for i in range(len(value)):
            if type(value[i]) is str:
                value[i] = sources.Images(value[i])

        # B. If some sources were removed, remove the associated spots. Also, avoid duplicates.
        if dadict.__contains__(self, 'sources') == True and len(self.sources) != 0:
            value = set(value)
            diff = value.symmetric_difference(self.__getprop__('sources'))
            remove = diff.difference(value)
            if len(remove) != 0:
                list.__init__(self, [c for c in self if c.source not in remove])
        self.__setprop__('sources', list(value))

    def process(self):
        """Process the sources to find content. Erase previous content."""
        list.__init__(self)
        p = self.processor
        p_args = p.func_code.co_varnames[1:p.func_code.co_argcount]
        for source in self.sources:
            data = p(source, **dict([(k, v) for k, v in self.iteritems() if k in p_args]))
            self.extend(data)
            self.update([(k, v) for k, v in data.iteritems() if k != 'sources'])

    def load(self, f):
        """
        Load a contents object from a file or string (pickle).

        Argument:
            f: file or string to load from.
        """
        if os.path.isfile(str(f)) is True:
            with open(f, 'r') as data:
                D = pickle.load(data)
        else:
            D = pickle.loads(f)
        list.__init__(self)
        self.update([i for i in D.__dict__.iteritems() if i[0] != '_attribs'], D)
        self._defaults = self._defaults
        self.extend(D)

    def __reduce__(self):
        """Handle advanced pickling."""
        return (type(self), tuple(), self.__dict__, iter([p for p in self]))

    def append(self, obj, switch_parenthood=True):
        """Add a spot and its source to the list."""
        try:
            if obj.source not in self.sources:
                self.sources.append(obj.source)
            if switch_parenthood == True:
                obj.parent = self
        except AttributeError:
            raise ValueError('Cannot append: invalid data type.')
        super(contents, self).append(obj)

    def extend(self, objs, switch_parenthood=False):
        """Add spots and their sources to the list."""
        try:
            for obj in objs:
                if obj.source not in self.sources:
                    self.sources.append(obj.source)
                if switch_parenthood == True:
                    obj.parent = self
        except AttributeError:
            raise ValueError('Cannot append: invalid data type.')
        super(contents, self).extend(objs)


@extensions.append
class content(np.ndarray):

    """
    More or less abstract class for any type of unary content.

    This class hijacks numpy ndarrays to access content. Every content object and subclass is a numpy.ndarray object and inherits all of its properties and goodness. On top of that, content objects allow for extra methods and attributes. Attributes will be conserved and reestablished during pickling/unpickling (or saving/loading) the object. Basically, this is an adapted record array.
    """

    def __array_finalize__(self, obj):
        """
        Set what numpy cannot set.

        Inherit the dict when slicing and co.
        Set the basal type as 'detection'
        """
        if obj is not None and '__dict__' in dir(obj):
            self.__dict__ = obj.__dict__

        if self.dtype.type != detection:
            self.dtype = np.dtype((detection, self.dtype))

    def __init__(self, *args, **kwargs):
        """Initiate the recarray."""
        super(content, self).__init__(args)

    def __reduce__(self):
        """Prepare the object for pickling. Inherit from the Numpy function and add the object's dictionary to the state."""
        func, args, state = np.recarray.__reduce__(self)
        state += (self.__dict__, )
        return (func, args, state)

    def __getitem__(self, item):
        """Return columns as ndarray, rather than content, and lines as detections."""
        i = super(content, self).__getitem__(item)
        if self.dtype.names and item in self.dtype.names:
            # What's weird is that recarray do not seem to need the second .view()... I don't understand all of it, though.
            return i.view(np.ndarray).view(self.dtype.fields[item][0])
        else:
            if type(i) == detection:
                i.parent = self
            return i

    def __getattr__(self, attr):
        """Support column access by attribute and fetching attributes in the parent classes."""
        if self.dtype.names and attr in self.dtype.names:
            return self[attr]
        elif getattr(self, 'parent', None) is not None:
            return getattr(self.parent, attr)
        else:
            raise AttributeError("'{0}' not found".format(attr))

    def __setattr__(self, attr, value):
        """Support column access by attribute."""
        if self.dtype.names and attr in self.dtype.names:
            self[attr] = value
        else:
            super(content, self).__setattr__(attr, value)

    def __setstate__(self, value):
        """Unpickle the object's state. Apply the state of the Numpy object and add the dictionary of the object."""
        super(content, self).__setstate__(value[0:5])
        self.__dict__ = value[5]


@extensions.append
class detection(np.void):
    """Sub-unit of content. A detection is one frame of the content."""

    def __getattr__(self, attr):
        """Support column access by attribute and fetching attributes in the base."""
        if self.dtype.names and attr in self.dtype.names:
            return self[attr]
        elif getattr(self, 'parent', None) is not None:
            return getattr(self.parent, attr)
        else:
            raise AttributeError("'{0}' not found".format(attr))

    def __setattr__(self, attr, value):
        """Support column access by attribute."""
        if self.dtype.names and attr in self.dtype.names:
            self[attr] = value
        else:
            super(detection, self).__setattr__(attr, value)

    def __getitem__(self, item):
        i = super(detection, self).__getitem__(item)
        # if item in ['x', 'y', 't'] or item in [0, 1, 5] and self.abs_position == True and isinstance(self.source, sources.ROI):
        #     conv = {0: 'x', 1: 'y', 5: 't'}
        #     if type(item) == int:
        #         item = conv[item]
        #     i += getattr(self.source, item).start
        return i

    def item(self):
        return tuple(self[k] for k in range(len(self.dtype.names)))

    def __repr__(self):
        return tuple(self[k] for k in range(len(self.dtype.names))).__repr__()
