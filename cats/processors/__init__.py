# -*- coding: utf8 -*-
"""
Processors transform images into numerical information.

>>> Do not believe the doc below. It is outdated. <<<

How to write a processor:
========================
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

"""

from types import ModuleType
import os
from glob import glob
import importlib

__all__ = []

for f in glob(os.path.dirname(os.path.realpath(__file__)) + '/[!_]*.py'):
    name = os.path.splitext(os.path.basename(f))[0]
    module = importlib.import_module('cats.processors.' + name)
    if '__processor__' in vars(module):
        for content, processors in module.__processor__.items():
            if content not in __all__:
                __all__.append(content)
                globals()[content] = ModuleType(content)
            for name, processor in processors.items():
                setattr(globals()[content], name, processor)

try:
    del f, module, name, content, processors, processor
except:
    pass
