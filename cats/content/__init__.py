# -*- coding: utf8 -*-
"""
Content is what CATS look for in images. It is the information written in the image that CATS translate in numerical form for you using processors (see the cats.processors module).

Different types of content can be found in images. This includes:
    - particles, like proteins, usually single gaussians
    - filaments, like DNA, usually a line of signal
    - barriers, that delimitate the experimental datasets within an image.
Types of content can be defined by user functions. For example, someone willing to obtain information on noise can write a function for the type of content 'noise'. Any name/content works, as long as you can write code to find it (see the documentation of the cats.processors module for more information on that.)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .. import extensions, defaults, sources
from ..adict import dadict
from copy import deepcopy
import numpy as np
import pickle
import os


@extensions.append
class Content(list, dadict):

    """
    More or less abstract class for any type of content.

    Content objects directly analyse sources upon binding. Hence, make sure to define the options and processors before you give the sources. During initialization of the object, the keyword arguments are first set, except 'sources', and then the sources are set. The object will automatically search content upon adding sources.

    Content classes inherit from lists and dadict. They contain dictionary items and list items. The __iter__ function (used in iterations, like in for loops) returns the content of the list (i.e.: the actual content), whereas the content of the dict (parameters, attributes, etc.) can be accessed via the usual dict methods items(), keys(), values() and their iter* equivalents.

    Example:
        c = Content(source1, source2, processor=processors.common, option1='value')
        func(*c)    # Unpacks the list
        func(**c)   # Unpacks the dict
    """

    def __init__(self, *args, **kwargs):
        """
        Initiate the object.

        During initialization of the object, the keyword arguments are first set, except 'sources', and then the sources are set from the non-keyword arguments and the keyword argument 'sources'. The object will automatically search content upon adding sources.

        Arguments:
            Paths to the sources or source object (cats.sources.Images or cats.sources.ROI). Paths will be transformed into cats.sources.Images objects.

        Keyword arguments:
            - sources: list of sources (cats.sources.ROI, cats.sources.Images or paths to the source.) If only one source, it can be given as is, without puting it in a list.
            - processor: Mandatory, unless set it the defaults (Content objects inherit from cats.adict.dadict,) The processor (see doc from the cats.processors module) to use to obtain the particles.
            - any argument for the processor to use
            - whatever you want, really.
        """
        dadict.__init__(self)
        if self.__class__.__name__ in dir(defaults):
            self._defaults = getattr(defaults, self.__class__.__name__)
        for k, v in kwargs.items():
            if k != 'sources':
                setattr(self, k, v)
        if 'sources' in kwargs:
            sources = kwargs['sources']
        elif len(args) != 0:
            sources = args
        else:
            sources = []
        self.sources = sources

    def __repr__(self):
        """Mix representations from list and dadict."""
        return list.__repr__(self) + '\n' + dadict.__repr__(self)

    def __getitem__(self, item):
        """Orchestrate items between list and dict."""
        if type(item) is int:
            return list.__getitem__(self, item)
        else:
            return dadict.__getitem__(self, item)

    def copy(self):
        """Copy the list and dict parts of the object."""
        r = type(self)(self.__dict__.copy())
        r.append([i for i in self])
        return r

    def deepcopy(self):
        """Copy the list and dict parts of the object without leaving references."""
        return deepcopy(self)

    @property
    def sources(self):
        """
        List of sources.

        Arguments:
            list of sources as the following types:
                paths to images (interpretable by cats.sources.Images),
                cats.sources.Images
                cats.sources.ROI
            All other input will lead to errors.
            If a single source is used, it does not need to be given as a list.

        A new source is processed into content upon addition to the list of sources. The content of a source that is removed is simultaneously deleted.
        """
        return self.__getprop__('sources')

    @sources.setter
    def sources(self, value):
        """Add/Set sources."""
        try:
            if type(value) is str or isinstance(value, sources.Images):
                value = [value]
            for i in range(len(value)):
                if type(value[i]) is str:
                    value[i] = sources.Images(value[i])
            value = set(value)  # Also, elimates doublons
            diff = value.symmetric_difference(self.__getprop__('sources'))
            add = diff.difference(self.__getprop__('sources'))
            remove = diff.difference(value)
            if len(remove) != 0:
                list.__init__(self, [c for c in self if c.source not in remove])
        except AttributeError:
            add = value
        # for source in add:
        #     self.extend(self.processor(source, **dict([(k, v) for k, v in self.items() if k in self.processor_args])))
        self.__setprop__('sources', list(value))

    @property
    def processor(self):
        """
        The function that extracts/transforms/processes images into content.

        Processors can be found in the module cats.processors.
        """
        return self.__getprop__('processor')

    @processor.setter
    def processor(self, value):
        """
        Set the processor and the processor_args attributes.
        Assumptions:
            - the processor is a function
            - the first argument is the source, all others are keywords arguments to be filled in by the object.
        """
        try:
            self.processor_args = value.func_code.co_varnames[1:value.func_code.co_argcount]
            self.__setprop__('processor', value)
        except AttributeError:
            raise ValueError('The selected processor is not a valid function')

    def process(self):
        """Process the sources to find content. Erase previous content."""
        list.__init__(self)
        for source in self.sources:
            self.extend(self.processor(source, **dict([(k, v) for k, v in self.items() if k in self.processor_args])))

    def load(self, f):
        """
        Load a Content object from a file or string (pickle).

        Argument:
            f: file or string to load from.
        """
        if os.path.isfile(str(f)) is True:
            with open(f, 'r') as data:
                D = pickle.load(data)
        else:
            D = pickle.loads(f)
        self.update([i for i in D.__dict__.iteritems() if i[0] != '_attribs'], D)
        self._defaults = self._defaults
        self.extend(D)
