# -*- coding: utf8 -*-
"""
Wrap all extensions into one module.

An extension is a function/module bound to a specific class as its method.

All files (modules) in the extensions subpackage must contain an '__extension__' dict that contain the name of the target class(es) (as given by class.__name__) as a key/keys, the value itself being a dictionary with the name of the method(s) as key(s) and the extensions' function as value. For example:
    __extension__ = {'Experiment': {'method': func}}
    __extension__ = {'Experiment': {'method': func}, 'Dataset': {'method2': func2}}
    __extension__ = {'Experiment': {'method': func}, 'Dataset': {'method': func}}
Where 'method' is the name of the function as called from the object, and 'func', the function in the extension's file.
"""

import os
from glob import glob
import importlib
from ..adict import adict

__all__ = ['append']

for f in glob(os.path.dirname(os.path.realpath(__file__)) + '/[!_]*.py'):
    name = os.path.splitext(os.path.basename(f))[0]
    module = importlib.import_module('cats.extensions.' + name)
    for Class, methods in module.__extension__.items():
        if Class not in __all__:
            __all__.append(Class)
            globals()[Class] = adict()
        for method, func in methods.items():
            globals()[Class][method] = func

    del f, module, name, Class, methods, method, func


def append(Class):
    """Decorate a class with its extensions."""
    if Class.__name__ in globals():
        for name, func in globals()[Class.__name__].items():
            setattr(Class, name, func)
    return Class
