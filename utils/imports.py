# -*- coding: utf8 -*-
"""Ease the import of submodules."""
import os
import glob
import importlib


def import_submodules(module=None, load=True):
    """
    List and/or load the submodules of the current.

    module: the path to the module or a subfile. __file__, usually.
    load (bool): load the submodules, if True. If False, simply list them.
    """
    mod_dir = os.path.dirname(os.path.realpath(module))
    mod_name, names = os.path.split(mod_dir)[1], list()
    for f in glob.glob(mod_dir + '/[!_]*.py'):
        name = os.path.splitext(os.path.basename(f))[0]
        names.append('cats.' + mod_name + '.' + name)
    if load == True:
        return dict((m, importlib.import_module(m)) for m in names)
    else:
        return names
