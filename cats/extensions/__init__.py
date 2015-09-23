"""
Wrap all extensions into one module.

Extensions are methods that are bound to a specific class as methods of this class.

All files (methods) in the extensions subpackage must contain a '__cats__' dict that contain:
        'target': the name of the class to append the extension to. List, if several targets.
        'method': (optional) the name of the method to call the extension with in the class. By default, it is the name of the module (file).
        'func': (optional) the name of the function, in this file, to bind to the class. By default, it is the function with the same name as the module (file).
"""

import os
from glob import glob
import importlib

__all__ = list()
for f in glob(os.path.dirname(os.path.realpath(__file__)) + '/[!_]*.py'):
    name = os.path.splitext(os.path.basename(f))[0]
    module = importlib.import_module('CATS.extensions.' + name)
    for func in module.__cats__:
        __all__.append(func)
        globals()[func] = getattr(module, func)
    if '__detection__' in dir(module):
        globals()[name] = getattr(module, module.__detection__)

del f, module, name, func
