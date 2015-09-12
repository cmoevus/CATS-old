#!env /usr/bin/python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
from copy import deepcopy


class AttributeDict(object):
    """
    Dictionary-like object that allow to access items as an object's attributes.

    AttributeDicts allow to add getters and setters on items, as well as to
    access them through AttributeDict.key or AttributeDict['key']. It can be
    used as a dictionary for enumerating items (as in **kwargs or in for
    loops). It supports hidden items (keys starting with '_') that do not
    appear in the item listings.

     How to write getters/setters:

    In order to keep keywords logical between the dictionary view and the
    attribute view of the object, when setting, getting or deleting a property,
    use self.__setprop__ (instead of object.__setattr__()), self.__getprop__()
    and self.__delprop__(), respectively. Indeed, object.__setattr__() will lead
    to infinite recursion and a 'transitive' name, (for example, '_prop' for
    'prop') will lead to loss of integrity between the dict view and the attrib
    view,  as when iterating over the values.

    Arguments:

    Any iterable object that can be accessed by iteritems() or as a pair of
    (keyword, value). Acceptable arguments are:
    - Dictionaries
    - Objects with a iteritems() function
    - Tuples/Lists of (keywork, value)
    - (keyword, value) as a tuple/list
    - keyworded arguments
    """

    fetched_functs = ['keys', 'values', 'items', 'iterkeys', 'iteritems',
                      'itervalues', '__iter__']

    def __init__(self, *args, **kwargs):
        self.__dict__['_attribs'] = dict()
        self.update(*args, **kwargs)

    def __getattr__(self, attr):
        """Return a fetched function, an attribute-item or an error."""
        error = '{0} not found'.format(attr)
        if attr in self.fetched_functs:
            return getattr(self.__dict__['_attribs'], attr)
        else:
            try:
                return self.__dict__['_attribs'][attr]
            except KeyError:
                raise AttributeError(error)

    def __setattr__(self, attr, value):
        """
        Put the right attribute/item into the right dict.

        Also:
        - Prevents from overwriting fetched functions
        - Transforms all subdicts into objects of the same type as this one
        """
        if attr in self.fetched_functs:
            # Prevent from blocking the read-only attributes from __dict__
            e = 'object attribute {0} is read-only'.format(attr)
            raise AttributeError(e)
        else:
            # Transform all subdicts into same type as self
            if type(value) is dict:
                value = type(self)(**value)

            # Put item in the right dict
            if attr[0] == '_' or attr in dir(self):
                object.__setattr__(self, attr, value)
            else:
                self.__dict__['_attribs'][attr] = value

    def __repr__(self):
        return self.__dict__['_attribs'].__repr__()

    def __str__(self):
        return self.__dict__['_attribs'].__str__()

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
            return key in self.__dict__['_attribs']

    def __delattr__(self, attr):
        try:
            object._delattr__(self, attr)
        except AttributeError:
            del self.__dict__['_attribs'][attr]

    def __setprop__(self, prop, value):
        """Set the value of a property. To be used in property setters instead of object.__setattr__()."""
        self.__dict__['_attribs'][prop] = value

    def __getprop__(self, prop):
        """Return the value of a property. To be used in property getters"""
        try:
            return self.__dict__['attrib'][prop]
        except KeyError:
            raise AttributeError

    def __delprop__(self, prop):
        """Delete a property. To be used in property deleters"""
        try:
            del self.__dict__['attrib'][prop]
        except KeyError:
            raise AttributeError

    def update(self, *args, **kwargs):
        """
        Update the object with the given list/object.
        """
        # Import dicts, lists and pairs
        for d in args:
            if 'iteritems' in dir(d) or isinstance(d, AttributeDict):
                d = d.iteritems()
            elif '__iter__' not in dir(d[0]):
                d = (d, )
            for k, v in d:
                setattr(self, k, v)
        # Import keywords
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def copy(self):
        return type(self)(self.__dict__.copy())

    def deepcopy(self):
        d = deepcopy(self.__dict__)
        return type(self)(d)
