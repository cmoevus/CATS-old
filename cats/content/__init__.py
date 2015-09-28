# -*- coding: utf8 -*-
"""
Content is what CATS look for in images. It is the information written in the image that CATS translate in numerical form for you.

Content can be accessed in the Experiment and Dataset objects (from the "experiment" submodule) by the methods "find", "get" and "filter".
    - find: search (and maybe find) the given type of content
    - get: return the given type of content (find it if if was not searched for yet)
    - filter: apply the given filter(s) on content
Additional methods to handle content can be created using extensions.

Different types of content can be found in images. This includes:
    - particles, like proteins, usually single gaussians
    - filaments, like DNA, usually a line of signal
    - barriers, that delimitate the experimental datasets within an image.
Types of content can be defined by user functions. For example, someone willing to obtain information on noise can write a function for the type of content 'noise'. Any name/content works, as long as you can write code to find it. See the howtos below.

How to write a content module:
=============================
- Write functions as object methods. The first argument will be the 'self'of the object calling the function.
- The 'find' functions are in charge of writing the content in 'self'. They should write the content as well as the parameters used to find the content. See "How to write a content-processing method".
- For CATS to be able to read and use the module, write a '__content__' dict as follow:
    - The key(s) of the dict is/are the name(s) of the type of content targeted by the module.
    - The value(s) is/are a dict contaning, as a key/keys, the name of the method that calls the function ('find', 'get', 'filter' or other custom methods from extensions) and as values, another dict contaning, as key(s), the name of the method and, as value, the function to call from the module
  For example:
    __content__ = {'particles': {'find': {'method': func}, 'get': {'method': func2}}}
    __content__ = {'particles': {'find': {'method': func}}, 'dna': {'find': {'method': func}}}
    __content__ = {'noise': {'find': {'method': func, 'method2': func2, 'filter': 'method3': func3}}}
 Where 'method' is the name of the function as known to the 'content' module, and 'func', the function in the extension's file.

How to write a content-processing method:
========================================
A method to process content is in charge of everything:
    - Getting the arguments from both keywords and object's parameters
    - Dealing with the content
    - Writing the result/content/information in the object
The advantage of this approach is flexibily. You can go and search any kind of information, as well as write it, anywhere in the object. This can be dangerous, but I trust you.
To increase interoperability between content modules and to keep a 'cats' logic, follow these directives:
    - Search for parameters in self['name of your content'], on top of kwargs
    - Do not allow general parameters (self.max_processes, for example), to be given as keywords. This will prevent fragmentation between modules.
    - Write the parameters used to find the content in the object. For example, if the method needs 'param1' and 'param2', write their value in self['name of your content']['param1'] and self['name of your content']['param2'].
    - Only write down these parameters AFTER finding the content. This will prevent fragmentation, if the search process lead to an exception, for example.
    - Write the unfiltered (from 'find') content in self['name of your content']['raw'].
    - Write the filtered content (from 'filter') in self['name of your content']['filtered']
    - Follow the directives for the specific types of content listed below.

How to write methods for particles:
==================================
These directives (on top of the other general Howtos) will make your module usable:
    - Write the list of particles as follow:
        - Each particle is a numpy recarray.
        - Each line in the recarray is one spot (call it 'detection' or 'gaussian'. It is what is on one frame).
        - Minimally, each particle must have the following columns:
            - 'x': (numerical) the position in x
            - 'y': (numerical) the position in y
            - 't': (int) the frame number
        - You can add as many fields as you want. The order does not matter, and this is the interest of the recarray versus lists. Here are a few standardized column names for particles:
            - 'i': intensity at the center of the spot
            - 's': the sigma of the gaussian of the spot

"""

import os
from glob import glob
import importlib
from ..adict import adict

__all__ = []

for f in glob(os.path.dirname(os.path.realpath(__file__)) + '/[!_]*.py'):
    name = os.path.splitext(os.path.basename(f))[0]
    module = importlib.import_module('cats.content.' + name)
    for content, access_methods in module.__content__.items():
        if content not in __all__:
            __all__.append(content)
            globals()[content] = adict()
        for access_method, methods in access_methods.items():
            if access_method not in globals()[content]:
                globals()[content][access_method] = adict()
            for method, func in methods.items():
                globals()[content][access_method][method] = func

    del f, module, name, content, access_methods, access_method, methods, method, func
