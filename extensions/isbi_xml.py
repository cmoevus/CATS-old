# -*- coding: utf8 -*-
"""Export a list of Particles in XML compatible with the ISBI Particle Tracking Challenge 2012 (Chenouard et al, 2014, Nature)."""
from __future__ import absolute_import, division, print_function
import xml.etree.ElementTree as ET


def as_isbi_xml(self, f='result.xml', snr=7, density='low med high 100.0', scenario='NO_SCENARIO'):
    """Return tracks as ISBI 2012 Challenge XML for scoring."""
    root = ET.Element('root')
    t = ET.SubElement(root, 'TrackContestISBI2012')
    t.attrib = {'SNR': str(snr), 'density': density, 'scenario': scenario}
    for particle in self:
        p = ET.SubElement(t, 'particle')
        for d in particle:
            s = ET.SubElement(p, 'detection')
            s.attrib = {'t': str(d['t']),
                        'x': str(round(d['x'], 3)),
                        'y': str(round(d['y'], 3)),
                        'z': '0'}
    E = ET.ElementTree(element=root)
    E.write(f)

__extension__ = {'Particles': {'as_isbi_xml': as_isbi_xml}}
