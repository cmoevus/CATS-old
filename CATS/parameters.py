#!env /usr/bin/python
# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
from copy import deepcopy
from CATS.attrdict import AttrDict
__all__ = ['Parameters']


class Parameters(AttrDict):

    """
    List of parameters accessible via AttrDict properties that also support default values.

    Parameters starting with '_' will be hidden. These can be Parameters' parameters or voluntarly hidden parameters. They have to be explicitely called by the user. They will not be listed by update() or items()/keys()/values()/etc.

    Keyword arguments:
        _defaults: (dict/AttrDict) the object to fetch values from if they do not exist in this object.
    """

    def __init__(self, *args, **kwargs):
        """Call parent class. Add support for defaults."""
        self._defaults = AttrDict() if '_defaults' not in kwargs else kwargs['_defaults']
        AttrDict.__init__(self, *args, **kwargs)

    def __getattr__(self, attr):
        """Call parent class. Add support for defaults."""
        error = '{0} not found'.format(attr)
        try:
            return AttrDict.__getattr__(self, attr)
        except AttributeError:
            try:
                if attr[0] == '_':
                    raise KeyError
                return self._defaults[attr] if not isinstance(self._defaults[attr], AttrDict) else self._defaults[attr].copy()
            except KeyError:
                raise AttributeError(error)

    def __setattr__(self, attr, value):
        """Call parent class. Add support for defaults."""
        # Transform all subdicts into same class, with proper defaults
        if attr != '_defaults' and type(value) is dict:
            d = self._defaults[attr] if attr in self._defaults else AttrDict()
            AttrDict.__setattr__(self, attr, Parameters(_defaults=d, **value))
        else:
            AttrDict.__setattr__(self, attr, value)

    def __getprop__(self, prop):
        """Call parent class. Add support for defaults."""
        try:
            return AttrDict.__getprop__(self, prop)
        except AttributeError:
            if prop in self._defaults:
                return self._defaults[prop]
            else:
                raise AttributeError('{0} not found.'.format(prop))

    def copy(self):
        """Copy to Parameters object rather than whatever subclass."""
        return Parameters(**self.__dict__.copy())

    def deepcopy(self):
        """Deep copy to Parameters object rather than whatever subclass."""
        return Parameters(**deepcopy(self.__dict__))
