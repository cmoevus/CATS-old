import xml.etree.ElementTree as ET


def as_isbi_xml(self, f='result.xml', snr=7, density='low med high 100.0', scenario='NO_SCENARIO'):
    """Return tracks as ISBI 2012 Challenge XML for scoring."""
    root = ET.Element('root')
    t = ET.SubElement(root, 'TrackContestISBI2012')
    t.attrib = {'SNR': str(snr), 'density': density, 'scenario': scenario}
    for track in self.get_tracks():
        p = ET.SubElement(t, 'particle')
        for spot in track:
            s = ET.SubElement(p, 'detection')
            s.attrib = {'t': str(spot['t']),
                        'x': str(round(spot['x'], 2)),
                        'y': str(round(spot['y'], 2)),
                        'z': '0'}
    E = ET.ElementTree(element=root)
    E.write(f)

__extension__ = {'Dataset': {'as_isbi_xml': as_isbi_xml}}
