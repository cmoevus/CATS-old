import numpy as np


def by_length_filter(self, parameters=None):
    """
    Filter the tracks obtained by link_spots based on parameters set either in self.filtration.by_length, either passed as arguments.

    Argument
        parameters: dict. Parameters to use for filtering. If None, will use the ones from the Dataset's parameters (Dataset.filtration).
    """
    if parameters is None:
        parameters = self.filtration
    if parameters['max_length'] is None:
        parameters['max_length'] = self.source.length
    n_tracks = list()
    for track in self.tracks:
        if parameters['min_length'] <= len(track) <= parameters['max_length']:
            if np.abs(np.diff(sorted([self.spots[s]['t'] for s in track]))).mean() <= parameters['mean_blink']:
                n_tracks.append(track)

    return n_tracks

__content__ = {'particles': {'filter': {'by_length': by_length_filter}}}
