"""
Wrap all extensions into one module.

An extension is a function/module bound to a specific class as its method.

All files (modules) in the extensions subpackage must contain a '__cats__' dict that contain:
    'target': the name of the class (as given by class.__name__) to append the extension to. List, if several targets.
    'method': (optional) the name of the method to call the extension with in the class. By default, it is the name of the module (file).
    'func': (optional) the name of the function, in this file, to bind to the class. By default, it is the function with the same name as the module (file).
"""

import os
from glob import glob
import importlib
from ..adict import adict

__all__ = ['append']

for f in glob(os.path.dirname(os.path.realpath(__file__)) + '/[!_]*.py'):
    name = os.path.splitext(os.path.basename(f))[0]
    module = importlib.import_module('cats.extensions.' + name)
    if 'method' in module.__cats__:
        name = module.__cats__['method']
    func = name if 'func' not in module.__cats__ else module.__cats__['func']
    if type(module.__cats__['target']) is str:
        module.__cats__['target'] = [module.__cats__['target']]
    for target in module.__cats__['target']:
        if target not in __all__:
            __all__.append(target)
            globals()[target] = adict()
        globals()[target][name] = getattr(module, func)
    del f, module, name, target, func


def append(Class):
    """Decorate a class with its extensions."""
    if Class.__name__ in globals():
        for name, func in globals()[Class.__name__].items():
            setattr(Class, name, func)
    return Class
