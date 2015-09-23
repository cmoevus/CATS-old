"""
Wrap all linkage methods into one module.

All files (methods) in the linkage subpackage must follow a specific structure to be properly read:
    1. The file must contain a __all__ list with the name of the objects to import in the module.
    2. If the name of the module (which is the name of the method) is not the name of the function used for linkage, the module must contain a __linkage__ string with the name of the function in it. The name of the module will be linked to that function in the 'linkage' module.
"""

import os
from glob import glob
import importlib

__all__ = list()
for f in glob(os.path.dirname(os.path.realpath(__file__)) + '/[!_]*.py'):
    name = os.path.splitext(os.path.basename(f))[0]
    module = importlib.import_module('CATS.linkage.' + name)
    for func in module.__all__:
        __all__.append(func)
        globals()[func] = getattr(module, func)
    if '__linkage__' in dir(module):
        globals()[name] = getattr(module, module.__linkage__)

del f, module, name, func
