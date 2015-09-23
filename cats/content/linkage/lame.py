from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import networkx as nx
from sys import stdout
import numpy as np
from numpy import ma
__all__ = ['lame_linkage']
__linkage__ = __all__[0]


def lame_linkage(self, verbose):
    # Reorganize spots by frame and prepare the Graph
    G = nx.DiGraph()
    n_frames = self.source.length
    frames = [[] for f in range(n_frames)]
    for i, spot in enumerate(self.spots):
        frames[spot['t']].append((spot['x'], spot['y'], i))
        G.add_node(i, frame=spot['t'])

    # Make optimal pairs for all acceptable frame intervals
    # (as defined in max_blink)
    for delta in range(1, self.linkage.max_blink + 1):
        if verbose is True:
            print('\rDelta frames: {0}'.format(delta), end='')
            stdout.flush()
        for f in range(n_frames - delta):
            # Matrix of distances between spots
            d = np.abs(np.array(frames[f])[:, np.newaxis, :2] -
                       np.array(frames[f + delta])[:, :2])

            # Filter out the spots with distances that excess max_disp in x and/or y
            disp_filter = d - self.linkage.max_disp >= 0
            disp_mask = np.logical_or(disp_filter[:, :, 0], disp_filter[:, :, 1])

            # Reduce to one dimension
            d = np.sqrt(d[:, :, 0]**2 + d[:, :, 1]**2)

            # Find the optimal pairs
            f_best = d != np.min(d, axis=0)
            fd_best = (d.T != np.min(d, axis=1)).T
            mask = np.logical_or(f_best, fd_best)
            mask = np.logical_or(mask, disp_mask)
            pairs = ma.array(d, mask=mask)

            # Organize in pairs, or edges, for graph purposes
            for s1, s2 in np.array(np.where(pairs.mask == False)).T:
                G.add_edge(frames[f][s1][2], frames[f + delta][s2][2],
                           weight=d[s1][s2])

    # Only keep the tracks that are not ambiguous (1 spot per frame max)
    tracks, a_tracks = list(), list()
    for track in nx.weakly_connected_component_subgraphs(G):
        t_frames = np.array(sorted([(s, self.spots[s]['t']) for s in track], key=lambda a: a[1]))

        # Good tracks
        if len(t_frames[:, 1]) == len(set(t_frames[:, 1])):
            tracks.append(sorted(track.nodes(), key=lambda s: self.spots[s]['t']))

        # Ambiguous tracks
        # This is a work in progress. More or less abandoned.
        elif self.linkage.ambiguous_tracks == True:
            nodes = track.nodes()
            track = dict([(self.spots[s]['t'], []) for s in nodes])
            for s in nodes:
                track[self.spots[s]['t']].append(s)
            ts = sorted(track.keys())
            for t in ts[:-1]:
                if len(track[t]) > 1:
                    now = track[t]
                    t_after = ts.index(t) + 1
                    after = track[ts[t_after]]
                    scores = np.abs(np.array([[self.spots[s]['x'], self.spots[s]['y']] for s in now])[:, np.newaxis] - np.array([[self.spots[s]['x'], self.spots[s]['y']] for s in after]))
                    scores = scores[:, :, 0] + scores[:, :, 1]
                    pair = np.where(scores == scores.min())
                    if len(pair[0]) > 1:
                        pair = (np.array(pair).T)[0]
                    now, after = [now[pair[0]]], [after[pair[1]]]
            track = sorted([t[0] for t in track.values()], key=lambda a: self.spots[a]['t'])
            ts = [self.spots[s]['t'] for s in track]
            a_tracks.append(track)

    if self.linkage.ambiguous_tracks == True:
        tracks.extend(a_tracks)

    if verbose is True:
        print('\nFound {0} tracks'.format(len(tracks)))
    return tracks
