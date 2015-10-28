# -*- coding: utf8 -*-
"""
Content is what CATS look for in images. It is the information written in the image that CATS translate in numerical form for you using processors (see the cats.processors module).

Different types of content can be found in images. This includes:
    - particles, like proteins, usually single gaussians
    - filaments, like DNA, usually a line of signal
    - barriers, that delimitate the experimental datasets within an image.
Types of content can be defined by user functions. For example, someone willing to obtain information on noise can write a function for the type of content 'noise'. Any name/content works, as long as you can write code to find it (see the documentation of the cats.processors module for more information on that.)

"""

import os
import glob
import importlib

# Import all submodules
__all__ = []
for f in glob.glob(os.path.dirname(os.path.realpath(__file__)) + '/[!_]*.py'):
    name = os.path.splitext(os.path.basename(f))[0]
    __all__.append('cats.content.' + name)
    module = importlib.import_module('cats.content.' + name)
print(__all__)
