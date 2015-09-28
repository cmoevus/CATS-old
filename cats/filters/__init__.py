import os
from glob import glob
import importlib

__all__ = list()
for f in glob(os.path.dirname(os.path.realpath(__file__)) + '/[!_]*.py'):
    name = os.path.splitext(os.path.basename(f))[0]
    module = importlib.import_module('cats.filters.' + name)
    for func in module.__all__:
        __all__.append(func)
        globals()[func] = getattr(module, func)

del f, module, name
