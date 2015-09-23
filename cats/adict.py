#!env /usr/bin/python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from copy import deepcopy
__all__ = ['adict', 'dadict']


class adict(object):

    """
    Dictionary-like object that allow to access items as an object's attributes.

    adicts allow to add getters and setters on items, as well as to access them through adict.key or adict['key']. It can be used as a dictionary for enumerating items (as in **kwargs or in for loops). It supports hidden items (keys starting with '_') that do not appear in the item listings.

    How to write getters/setters:

    In order to keep keywords logical between the dictionary view and the
    attribute view of the object, when setting, getting or deleting a property,
    use self.__setprop__ (instead of object.__setattr__()), self.__getprop__()
    and self.__delprop__(), respectively. Indeed, object.__setattr__() will lead
    to infinite recursion and a 'transitive' name, (for example, '_prop' for
    'prop') will lead to loss of integrity between the dict view and the attrib
    view,  as when iterating over the values.

    Methods:

    update: just like dict()'s update.
    copy: just like dict()'s copy.
    deepcopy: copy recursively the object so that no reference is left.

    Arguments:

    Any iterable object that can be accessed by iteritems() or as a pair of
    (keyword, value). Acceptable arguments are:
        Dictionaries
        Objects with a iteritems() function
        Tuples/Lists of (keywork, value)
        (keyword, value) as a tuple/list
        keyworded arguments
    """

    fetched_functs = ['keys', 'values', 'items', 'iterkeys', 'iteritems',
                      'itervalues', '__iter__']

    def __init__(self, *args, **kwargs):
        """Set up the attribute dict and values."""
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

    def __delattr__(self, attr):
        """Delete attribute, hidden or not."""
        try:
            object._delattr__(self, attr)
        except AttributeError:
            del self.__dict__['_attribs'][attr]

    def __repr__(self):
        """Return the representation of the attribute dict."""
        return self.__dict__['_attribs'].__repr__()

    def __str__(self):
        """Return the string representation of the attribute dict."""
        return self.__dict__['_attribs'].__str__()

    def __setitem__(self, key, value):
        """Consider items as attributes."""
        self.__setattr__(key, value)

    def __getitem__(self, key):
        """Consider items as attributes."""
        try:
            return self.__getattribute__(key)
        except AttributeError:
            return self.__getattr__(key)

    def __contains__(self, key):
        """Look if the adict has an item, hidden or not."""
        if key[0] == '_':
            return key in self.__dict__
        else:
            return key in self.__dict__['_attribs']

    def __setprop__(self, prop, value):
        """Set the value of a property. To be used in property setters instead of object.__setattr__()."""
        self.__dict__['_attribs'][prop] = value

    def __getprop__(self, prop):
        """Return the value of a property. To be used in property getters."""
        try:
            return self.__dict__['_attribs'][prop]
        except KeyError:
            raise AttributeError

    def __delprop__(self, prop):
        """Delete a property. To be used in property deleters."""
        try:
            del self.__dict__['_attribs'][prop]
        except KeyError:
            raise AttributeError

    def update(self, *args, **kwargs):
        """Update the object with the given list/object."""
        # Import dicts, lists and pairs
        for d in args:
            if 'iteritems' in dir(d) or isinstance(d, adict):
                d = d.iteritems()
            elif '__iter__' not in dir(d[0]):
                d = (d, )
            for k, v in d:
                setattr(self, k, v)
        # Import keywords
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def copy(self):
        """"Copy the adict object a la dict.copy."""
        return type(self)(self.__dict__.copy())

    def deepcopy(self):
        """"Recursively copy the adict object so that no reference to the original object is left."""
        d = deepcopy(self.__dict__)
        return type(self)(d)


class dadict(adict):

    """
    Attribute dictionary that also support default values.

    dadict's copy and deepcopy functions, unlike adict's, do not consider the subclasses' type. All classes inheriting copy and deepcopy from dadict will return a dadict object with copy() or deepcopy().

    Keyword arguments:
        _defaults: (dict/adict) the object to fetch values from if they do not exist in this object.
    """

    def __init__(self, *args, **kwargs):
        """Call parent class. Add support for defaults."""
        self._defaults = adict() if '_defaults' not in kwargs else kwargs['_defaults']
        adict.__init__(self, *args, **kwargs)

    def __getattr__(self, attr):
        """Call parent class. Add support for defaults."""
        error = '{0} not found'.format(attr)
        try:
            return adict.__getattr__(self, attr)
        except AttributeError:
            try:
                if attr[0] == '_':
                    raise KeyError
                return self._defaults[attr] if not isinstance(self._defaults[attr], adict) else self._defaults[attr].copy()
            except KeyError:
                raise AttributeError(error)

    def __setattr__(self, attr, value):
        """Call parent class. Add support for defaults."""
        # Transform all subdicts into same class, with proper defaults
        if attr != '_defaults' and type(value) is dict:
            d = self._defaults[attr] if attr in self._defaults else adict()
            adict.__setattr__(self, attr, dadict(_defaults=d, **value))
        else:
            adict.__setattr__(self, attr, value)

    def __getprop__(self, prop):
        """Call parent class. Add support for defaults."""
        try:
            return adict.__getprop__(self, prop)
        except AttributeError:
            if prop in self._defaults:
                return self._defaults[prop]
            else:
                raise AttributeError('{0} not found.'.format(prop))

    def copy(self):
        """Copy to dadict object rather than whatever subclass."""
        return dadict(**self.__dict__.copy())

    def deepcopy(self):
        """Deep copy to dadict object rather than whatever subclass."""
        return dadict(**deepcopy(self.__dict__))
