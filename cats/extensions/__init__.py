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

from types import ModuleType
import importlib
from cats.utils.imports import import_submodules

__all__ = ['append']

submodules = import_submodules(__file__)
__all__.extend(submodules.keys())
for submod in submodules.values():
    if '__extension__' in vars(submod):
        for Class, methods in submod.__extension__.items():
            if Class not in __all__:
                __all__.append(Class)
                globals()[Class] = ModuleType(Class)
            for method, func in methods.items():
                setattr(globals()[Class], method, func)
        del submod, Class, methods, method, func
del import_submodules, submodules


def append(Class):
    """Decorate a class with its extensions."""
    if Class.__name__ in globals():
        for name in dir(globals()[Class.__name__]):
            if name != '__doc__' and name != '__name__':
                setattr(Class, name, getattr(globals()[Class.__name__], name))
    return Class
