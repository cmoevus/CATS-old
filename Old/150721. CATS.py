#!env /usr/bin/python
 # -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
from scipy import ndimage, misc, signal, stats
from scipy.optimize import curve_fit
from skimage.feature import blob_log
from skimage.filters import threshold_otsu, gaussian_filter
from skimage.transform import rotate
from skimage import io
from time import time
import networkx as nx
from colorsys import hls_to_rgb
from random import randrange
from math import floor, ceil, atan, degrees
from itertools import product
import sys

class DNACurtain:
    '''
    A classic DNA curtain experiment
    '''

    # The magnification at which the experiment was done, in X (usually, 60 or 90)
    magnification = None

    # The type of patterns used for DNA curtains. Usually, 0
    # 0: Charlie Brown with pedestals
    # 1: Charlie Brown without pedestals
    patterns = None

    # Pixel size after magnification, in microns
    pixel_size = None

    # Pixel size on the camera, in microns. Usually, 16 microns
    camera_pixel = None

    # Initialize the object
    def __init__(self, params=True):

        # Default, common parameters
        if params is True:
            self.magnification = 60
            self.patterns = 0
            self.pixel_size = 0.2667
            self.camera_pixel = 16

    # Find the spots (blobs) in the image:
    #def find_spots(self):

# Barriers detection
def find_barriers(source, frames=(0, 100), ylims=False):
    '''
    Finds the barriers in an experiment.
    source: the source folder of the experiments' tif files
    frames: the frames to use for the z projection

    SHOULD ADD OPTION TO REJECT INCOMPLETE SETS OF BARRIERS
    NEED TO IMPLEMENT:
        - IMCOMPLETE BARRIER SETS
    '''

    # Variables
    # Values for Charlie Brown DT: 12.5um and 37um
    dbp = 47 # distance between barriers and pedestals (px)
    dbb = 139 # distance between barriers (px)
    approx = 20 # allowed error on the distances (px)
    blur = 2 # Stdev of the Gaussian kernel to use for smoothing the images
    guess_lost_barriers = True

    # Make a projection of the image
    files = sorted(os.listdir(source))
    frames = (0, len(files)) if frames is False else frames
    imgs = [io.imread(source+'/'+i) for i in files[slice(*frames)]]
    img = np.zeros(imgs[0].shape)
    for i in imgs:
        img += i
    img = gaussian_filter(img, blur)

    # Limits
    if ylims is False:
        ylims = (0, img.shape[0])
    else:
        ylims = (img.shape[0] - ylims[1], img.shape[0] - ylims[0])

    projection = np.zeros((img.shape[1]))
    for i, c in enumerate(img.T):
        projection[i] = sum(c[ylims[0]:ylims[1]])

    # Find peaks
    d_proj = np.diff(projection)
    tangents = list()
    for i in range(0, len(d_proj)-1):
        neighbors = sorted((d_proj[i], d_proj[i+1]))
        if neighbors[0] < 0 and neighbors[1] > 0:
            tangents.extend([i])
    extremas = np.where(np.square(np.diff(d_proj)) - np.square(d_proj[:-1]) >= 0)[0]
    peaks = sorted([p for p in tangents if p in extremas])
    distances = np.subtract.outer(peaks, peaks)

    # Build sets of barriers
    max_sets = int(ceil(len(projection) - dbp)/dbb) + 1
    exp_dists, sets = [dbp], list()
    for i in range(1, max_sets):
        exp_dists.extend([i*dbb - dbp, i*dbb])

    #Find the best possible match for consecutive barriers-pedestals for each peak
    for peak in range(distances.shape[0]):
        i = 0 # Position in the sets of barriers
        j, last_j = 0, -1 # Distance with last peak
        barriers_set, current_set = list(), list()
        a = 0 # Allowed difference for ideal distance
        while i < max_sets and j < len(exp_dists):

            # Update the distances to search at
            if j != last_j:
                search = [exp_dists[j]]
                for k in range(1, approx): 
                    search.extend([exp_dists[j] - k, exp_dists[j] + k])
                last_j = j

            # Try to find a pair
            match = np.where(distances[peak] == search[a])[0]
            if len(match) != 0:
                if j == 0:
                    current_set = (peaks[peak], peaks[match[0]])
                    barriers_set.append((current_set, a))
                else:
                    i += j // 2 + 1
                    if len(current_set) == 1:
                        barriers_set.append((current_set, approx + 1))
                    current_set = [match[0]]
                peak = match[0]
                j = 0 if j % 2 == 1 else 1
                a = 0

            # No pair found: look for it with more laxity
            else:
                a += 1

            # This pair does not exists in all permitted limits. Look for next pair.
            if a == approx:
                a = 0
                j += 1

        if len(barriers_set) > 0:
            sets.append(barriers_set)

    set_scores = list()
    for s in sets:
        score = sum([i[1] for i in s])
        set_scores.append((len(s), -score))
    barriers = sorted([sorted(i[0]) for i in sets[set_scores.index(sorted(set_scores, reverse=True)[0])]])

    return barriers

def correct_rotation(source, destination, bins=10):
    '''
    Corrects rotation on images based on the barriers
    Assumes that rotation is constant over time.

    source: the directory containing the images
    destination: where to write the corrected images
    bins: number of bins to split the image in, vertically (lower if noise is higher)
    '''

    images = sorted(os.listdir(source))
    shape = io.imread(source+'/'+images[0]).shape
    bins = np.linspace(0, shape[1], bins, dtype=int)
    barriers = list()
    for i in range(len(bins) - 1):
        #frames = (0, len(images)), 
        barriers.append([i for j in find_barriers(source, ylims=(bins[i], bins[i+1])) for i in j])

    m = list()
    for y in np.array(barriers).T:
        y -= y[0]
        m.append(np.polyfit(bins[1:], y, 1)[0])
    angle = degrees(atan(np.mean(m)))

    for i in images:
        io.imsave(destination+'/'+i, rotate(io.imread(source+'/'+i), angle, mode='nearest', preserve_range=True).astype(dtype=np.int16))

    return True

# Finds the spots in the images in a dir
def find_spots(images_source, blur, threshold, xlims = False, ylims = False):
    files = sorted(os.listdir(images_source))
    if xlims is False or ylims is False:
        image = io.imread(images_source+'/'+files[0])
        xlims = (0, image.shape[1]) if xlims is False else xlims
        ylims = (0, image.shape[0]) if ylims is False else ylims

    spots, i = list(), 0
    for f in files:
        image = io.imread(images_source+'/'+f)[ylims[0]:ylims[1], xlims[0]:xlims[1]]
        b = blob_log(ndimage.gaussian_filter(image, blur), threshold=threshold, min_sigma=1)
        spots.extend([(s[1] + xlims[0], s[0] + ylims[0], s[2], i) for s in b])

        print('Frame {0} done, {1} spots\r'.format(i, len(b)))
        i += 1

    return spots

def find_spots_between_barriers(source, blur, threshold, frames_barriers=False):
    '''
    Find exclusively the spots that are localized between the barriers
    'blur' and 'threshold' can be preoptimized using the 'test_spots_conditions' function.

    source: the directory containing the images
    blur: the blur to apply on the image for finding spots
    threshold: the threshold to use for finding spots
    frames_barriers: the frames to use for finding the barriers. If False, uses all frames.
    '''

    barriers = find_barriers(source, frames_barriers)
    spots = list()
    for barrier_set in barriers:
        spots.extend(find_spots(source, blur, threshold, xlims=barrier_set))

    return sorted(spots, key=lambda a: a[3])

# Tests blur and threshold conditions for finding spots
def test_spots_conditions(inp, outp, blur, threshold):
    t = time()
    # Find spots in given conditions
    image = io.imread(inp)
    b = blob_log(ndimage.gaussian_filter(image, blur), threshold=threshold, min_sigma=1)

    # Prepare output image
    shape = list(image.shape)
    ni = np.zeros([3] + shape)
    ni[...,:] = image/image.max()*255
    ni = ni.transpose(1, 2, 0)

    # Write spots
    for s in b:
        ni[s[0],s[1],:] = (255, 0, 0)
    misc.imsave(os.path.normpath(outp), ni)

    print('Found {0} spots in {1:.2f}s\r'.format(len(b), time()-t))
    return b

# Writes spots to a file
def write_spots(spots, f):
    with open(f, 'w') as record:
        for spot in spots:
            record.write('\t'.join([str(x) for x in spot])+'\n')

# Import spots
def get_spots(source, xlims = None, ylims = None, zlims = None):
    spots = list()
    with open(source, 'r') as blobs:
        if xlims != None and ylims != None and zlims != None:
            for x, y, s, t in [blob.split('\t') for blob in blobs]:
                x, y, s, t = int(x), int(y), int(s), int(t)
                if xlims[0] <= x <= xlims[1] and ylims[0] <= y <= ylims[1] and zlims[0] <= t <= zlims[1]:
                    spots.append((x, y, s, t))
        else:
            for x, y, s, t in [blob.split('\t') for blob in blobs]:
                spots.append((float(x), float(y), float(s), int(t)))

    return sorted(spots, key=lambda b: b[3])

# Refine the spots to subpixel resolution
def subpixel_resolution(spots, images_source):
    # Reorganize spots by frame
    n_frames = np.array(spots, dtype='i8')[:, 3].max() + 1
    frames = [[] for f in range(n_frames)]
    i = 0
    for spot in spots:
        frames[spot[3]].append(list(spot)+[i])
        i += 1

    # Fit a 2D gaussian on every single spot
    i = 0
    #r = np.sqrt(2)
    r=4
    spr = [[] for s in spots]
    for f in sorted(os.listdir(images_source)):
        image = io.imread(d+'/'+f)
        xlims = (0, image.shape[1])
        ylims = (0, image.shape[0])
        for s in frames[i]:
            y = [max(int(floor(s[1] - r*s[2])), ylims[0]), 
                 min(int(ceil(s[1] + r*s[2]))+1, ylims[1])]
            x = [max(int(floor(s[0] - r*s[2])), xlims[0]), 
                 min(int(ceil(s[0] + r*s[2]))+1, xlims[1])]
            data = image[y[0]:y[1],x[0]:x[1]]
            coords = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
            try:
                fit, cov = curve_fit(gaussian_2d, coords, data.ravel(), p0=(image[s[1]][s[0]], s[0]-x[0], s[1]-y[0], s[2]))
                print(s[:2], [fit[1]+x[0], y[0]+fit[2]], np.sqrt(np.diag(cov)))
                plt.imshow(data, cmap=plt.cm.gray, origin='bottom')
                plt.plot(fit[1], fit[2], 'ro')
                plt.plot(s[0]-x[0], s[1]-y[0], 'go')
                plt.show()
                s = [x[0]+fit[1], y[0]+fit[2], fit[3], i, s[4]]
            except:
                pass
            spr[s[4]] = s
            #print(spr[s[4]])
        i += 1

    return spr

def gaussian_2d(coords, A, x0, y0, s):
    x, y = coords
    return (A*np.exp(((x-x0)**2+(y-y0)**2)/(-2*s**2))).ravel()

# Make tracks out of continuous blobs over time
def make_tracks(spots):
    # Variables
    max_disp = [2, 2] # Maximum displacement, in pixels
    max_blink = 25 # Maximum blinking, in frames
    ambiguous_tracks = False # Should we keep and try to solve ambiguous tracks?

    # Reorganize spots by frame
    n_frames = np.array(spots, dtype=int)[:, 3].max() + 1
    frames = [[] for f in range(n_frames)]
    i = 0
    for spot in spots:
        frames[spot[3]].append(list(spot[:2])+[i])
        i += 1

    # Prepare the graph
    G = nx.DiGraph()
    [G.add_node(i, frame=n[3]) for i, n in enumerate(spots)]

    # Make optimal pairs for all acceptable frame intervals (as defined in max_blink)
    for delta in range(1, max_blink+1):

        print('Delta frames: {0}'.format(delta))
        for f in range(n_frames - delta):
            # Measure the distances between spots
            d = np.abs(np.array(frames[f])[:, np.newaxis,:2] - np.array(frames[f+delta])[:,:2])
 
            # Filter out the spots with distances that excess max_disp in x and/or y
            disp_filter = d - max_disp >= 0
            disp_mask = np.logical_or(disp_filter[:,:,0], disp_filter[:,:,1])

            # Reduce the 3rd dimension
            d = np.sqrt(d[:,:,0]**2+d[:,:,1]**2)

            # Find the optimal pairs
            f_best = d != np.min(d, axis=0)
            fd_best = (d.T != np.min(d, axis=1)).T
            mask = np.logical_or(f_best, fd_best)
            mask = np.logical_or(mask, disp_mask)
            pairs = ma.array(d, mask=mask)

            # Organize in pairs, or edges, for graph purposes
            for s1, s2 in np.array(np.where(pairs.mask == False)).T:
                G.add_edge(frames[f][s1][2], frames[f+delta][s2][2], weight=d[s1][s2])

    # Only keep the tracks that are not ambiguous (1 spot per frame max)
    tracks, a_tracks = list(), list()
    for track in nx.weakly_connected_component_subgraphs(G):
        t_frames = np.array(sorted([(s, spots[s][3]) for s in track], key=lambda a:a[1]))

        # Good tracks
        if len(t_frames[:, 1]) == len(set(t_frames[:, 1])):
            tracks.append(sorted(track.nodes(), key=lambda s: spots[s][3]))

        # Ambiguous tracks
        elif ambiguous_tracks is True:
            print(track.number_of_nodes(), track.number_of_edges())
            track = transitive_reduction(track, t_frames[:, 0])
            sources, junctions = list(), list()
            for i in track.in_degree().items():
                if i[1] == 0:
                    sources.append(i[0])
                elif i[1] > 1:
                    junctions.append(i[0])
            sinks = [i[0] for i in track.out_degree().items() if i[1] == 0]
            a_tracks.append(sorted(track.nodes(), key=lambda s: spots[s][3]))

            print(track.number_of_nodes(), track.number_of_edges())
            print(sources)
            print(junctions)
            print(sinks)
            a = np.array([(s, spots[s][3], spots[s][0], spots[s][1]) for s in sorted(track.nodes(), key=lambda a:spots[a][3])])
            print([i for i in nx.weakly_connected_components(track)])
            for b, c, d, e in a:
                print(b, '\t', c, '\t', d, '\t', e)
            sl, f, x, y = a.T
            plt.figure()
            plt.subplot(211)
            plt.scatter(x, y, c=f)
            plt.subplot(212)
            nx.draw_networkx(track)
            plt.show()

    if ambiguous_tracks == False:
        return tracks
    else:
        return tracks, a_tracks

# Filter tracks. Quality control
def filter_tracks(tracks, spots):
    max_blink = 25 # Maximum blinking, in frames
    mean_blink = 5 # Maximum mean blinking, in frames, per track
    min_length = 3 # Minimum number of data points in a track
    max_length = 9500

    n_tracks = list()
    for track in tracks:
        if  min_length <= len(track) <= max_length:
            if np.abs(np.diff(sorted([spots[s][3] for s in track]))).mean() <= mean_blink:
                n_tracks.append(track)

    return n_tracks

# Writes tracks to a file, to be reused with the same spots file and the same limits.
def write_tracks(tracks, f):
    with open(f, 'w') as record:
        for track in tracks:
            record.write(' '.join([str(x) for x in track])+'\n')

# Import tracks
def get_tracks(source):
    ts = list()
    with open(source, 'r') as tracks:
        for track in tracks:
            t = list()
            for s in track.split(' '):
                t.append(int(s))
            ts.append(t)

    return ts

# Draw tracks
def draw_tracks(tracks, spots, d, destination, draw_spots = False):
    n_frames = np.array(spots, dtype=int)[:, 3].max() + 1

    # Make colors
    colors = [np.array(hls_to_rgb(randrange(0, 360)/360, randrange(20, 80, 1)/100, randrange(20, 80, 10)/100))*255 for i in tracks]

    # Separate tracks per image, because memory is short
    frames = [[] for f in range(n_frames)]
    i = 0
    for track in tracks:
        for s in track:
            s = spots[s]
            spot = list(s[:2])
            spot.extend(colors[i])
            frames[s[3]].append(spot)
        i += 1

    # Draw all spots
    if draw_spots is True:
        used = set([spot for track in tracks for spot in track])
        unused = used.symmetric_difference(range(0, len(spots)))
        for s in unused:
            s = spots[s]
            spot = list(s[:2])
            spot.extend([255, 0, 0])
            frames[s[3]].append(spot)

    # Write tracks RGB images
    j = 0
    for f in sorted(os.listdir(d)):

        # Make images RGB
        i = io.imread(d+'/'+f)
        shape = list(i.shape)
        ni = np.zeros([3] + shape)
        ni[...,:] = i/i.max()*255
        ni = ni.transpose(1, 2, 0)

        # Write spots
        for s in frames[j]:
            ni[s[1],s[0],:] = s[2:]
        misc.imsave(os.path.normpath(destination)+'/'+str(j)+'.tif', ni)
        j += 1

# Draws a SY Plot
def sy_plot(tracks, spots, binsize=3):
    '''
    Draws a SY (Stacked Ys) plot based on the tracks
    '''

    # Define the limits in X
    lims = (0, max([spots[spot][0] for track in tracks for spot in track]))

    # Create a dic of position -> lengths of tracks
    bins = np.arange(0, lims[1], binsize)
    y = dict()
    for i in range(0, len(bins)):
        y[i] = list()

    n = 0
    for track in tracks:
        frames = [spots[spot][3] for spot in track]
        length = max(frames) - min(frames) + 1
        position = np.mean([spots[spot][0] for spot in track])
        b = max([i for i, b in enumerate(bins) if position - b >= 0])
        y[b].append(np.log(length))
        n += 1

    # Build a masked array to feed plt.pcolor with
    max_length = max([len(t) for t in y.values()])
    C = np.ma.zeros((len(bins), max_length))
    mask = np.ones((len(bins), max_length), dtype=np.bool)
    for x, data in y.items():
        datapoints = len(y[x])
        # Unmask 
        mask[x, 0:datapoints] = False
        # Fill the column
        C[x, 0:datapoints] = sorted(y[x])

    C.mask = mask
    plt.figure()
    plt.pcolor(C.T)
    cb = plt.colorbar()
    cb.set_label('Length of the interaction (frames)')
    lengths = [v for l in y.values() for v in l]
    cb_ticks = np.linspace(min(lengths), max(lengths), 10)
    cb.set_ticks(cb_ticks)
    cb.set_ticklabels([int(round(np.e**v, 0)) for v in cb_ticks])
    ticks = range(0, len(bins), 2)
    plt.xticks(ticks, [int(bins[i]) for i in ticks])
    plt.figtext(0.25, 0.87, ha='right', va='top', s='N={0}'.format(n))
    plt.xlabel('Position on DNA (pixel)')
    plt.ylabel('Number of interactions')
    plt.savefig('sy_plot.png')
    plt.close()

    return bins, y

# Draws a histogram of position on DNA
def histogram(tracks, spots, binsize=3, prop=1, use_tracks = True):
    lims = (0, max([spot[prop] for spot in spots]))
    bins = int(ceil((lims[1]  - lims[0])/binsize))

    if use_tracks is False:
        data = [spot[prop] for spot in spots]
    else:
        data = [np.mean([spots[spot][prop] for spot in track]) for track in tracks]

    plt.figure()
    plt.xlabel('Property {0} of the spot'.format(prop))
    plt.ylabel('Number of interactions')
    plt.hist(data, bins=bins)
    plt.savefig('histogram.png')
    plt.close()

# Zhi plot drawing function: dwell time vs initial binding time
def zhi_plot(tracks, spots):
    fps = 1/0.06
    binsize = int(ceil(1000/fps))
    exp_len = int(ceil(10000/fps))

    bins = [[] for t in range(0, exp_len, binsize)]
    for track in tracks:
        frames = [spots[s][3] for s in track]
        m = min(frames)
        dt = (max(frames) - m)/fps
        bins[int(floor((m/fps)/binsize))].append(dt)

    # All the dwell times
    xs = list()
    ys = list()
    for i, b in enumerate(bins):
        xs.extend([i*binsize for d in b])
        ys.extend(b)
    plt.plot(xs, ys, 'g.', label='Calculated dwell times')

    # Mean dwell tims
    y = [np.mean(b) for b in bins]
    x = range(0, exp_len, binsize)
    plt.plot(x, y, 'ro', label='Mean dwell time: {0:.2f}s'.format(np.mean(ys)))
    plt.ylabel('Mean dwell time (s)')
    plt.yscale('log')
    plt.xlabel('Binned time of initial binding (s)')
    plt.legend()
    plt.figtext(0.82, 0.12, 'N={0}'.format(len(tracks)))
    plt.savefig('150720. Dwell time vs initial binding time.png')

# Return the transitive reduction of a given graph
def transitive_reduction(G, order=None, adj_matrix=False):
    '''
    Returns the transitive reduction of a given graph
    Based on Aho, A. V., Garey, M. R., Ullman, J. D. The Transitive Reduction of a Directed Graph. SIAM Journal on Computing 1, 131–137 (1972).

    order: the order in which the vertices appear in time (to fit the second condition). If None, uses the same order as in the graph
    adj_matrix: returns an adjacency matrix if True, a Graph if False
    '''

    # Transitive closure
    MT = transitive_closure(G, order, True)

    # Reorganize the adjacency matrix
    V = G.nodes()
    M = nx.to_numpy_matrix(G, nodelist=order, weight=None)

    # Adjacency matrices operations
    Mt = np.array(M - M.dot(MT))

    # Return in the proper format
    if adj_matrix is True:
        return np.where(Mt != 1, 0, 1).astype(dtype=bool)
    else:
        Gt = G.copy()
        for i, j in np.vstack(np.where(Mt != np.array(M))).T:
            try:
                Gt.remove_edge(order[i], order[j])
            except:
                pass
        return Gt

# Returns the transitive closure of a given Graph
def transitive_closure(G, order=None, adj_matrix=False):
    '''
    This method assumes that your graph is:
        1. Directed acyclic
        2. Organized so that a node can only interact with a node positioned after it, in the adjacency matrix 

    order: the order in which the vertices appear in time (to fit the second condition)
    adj_matrix: returns an adjacency matrix if True, a Graph if False
    '''
    V = G.nodes()
    M = nx.to_numpy_matrix(G, nodelist=order, weight=None)

    # Close the graph
    for i, j in sorted((np.array(np.where(M==1), dtype=int).T)[0], key=lambda a:[0])[::-1]:
        M[i] = np.logical_or(M[i], M[j])

    # Return in the proper format
    if adj_matrix is True:
        return M
    else:
        GT = G.copy()
        for i, j in (np.array(np.where(M == 1)).T):
            GT.add_edge(order[i], order[j])
        return GT

# Output
if __name__ == '__main__':

    # Stuff to use: HR
    #spots_f = '/home/corentin/HR3rd.spots'
    #tracks_f = '/home/corentin/HR3rd.tracks'
    #d = '/home/corentin/HR nucleosome 565 3rd barrier/'
    #destination = '/home/corentin/HR nucleosome 565 3rd barrier results/'

    #spots = get_spots(spots_f, (2, 45), (5, 507), (0, 10000))
    #tracks = get_tracks(tracks_f)
    #tracks, a_tracks = make_tracks(spots)
    #draw_tracks(a_tracks, spots, d, destination)
    #tracks = filter_tracks(tracks, spots)
    #histogram(tracks, spots, 5, prop=0)
    #sy_plot(tracks, spots, 3)
    #zhi_plot(tracks, spots)

    ## Stuff to use: nucs
    #spots_f = '/home/corentin/H2AATTO532.spots'
    #tracks_f = '/home/corentin/H2AATTO532.tracks'
    d = '/home/corentin/H2A ATTO532 data/'
    #destination = '/home/corentin/H2A ATTO532 data results/'
    print(find_barriers(d))
    #t_inp = d+'/DT in rep buffer-10110.tif'
    #t_out = '/home/corentin/H2A ATTO532 test.tif'
    #spots = find_spots_between_barriers(d, 0.5, 0.005)

    #d = '/home/corentin/12Jul15fc1002565/'
    #destination = '/home/corentin/12Jul15fc1002565_corrected/'
    #correct_rotation(d, destination)

    #d = '/home/corentin/12Jul15fc1002565_corrected/'
    #destination = '/home/corentin/12Jul15fc1002565 data/'
    #spots_f = '/home/corentin/12Jul15fc1002565_corrected.spots'
    #tracks_f = '/home/corentin/12Jul15fc1002565_corrected.tracks'
    #spots = find_spots_between_barriers(d, 0.5, 0.001)
    #write_spots(spots, spots_f)
    #spots = get_spots(spots_f)
    #tracks = make_tracks(spots)
    #write_tracks(tracks, tracks_f)
    #tracks = get_tracks(tracks_f)
    #tracks = filter_tracks(tracks, spots)
    #draw_tracks(tracks, spots, d, destination)

    #spots = get_spots(spots_f)
    ##spr = subpixel_resolution(spots, d)
    #tracks = make_tracks(spots)
    ##write_tracks(tracks, tracks_f)
    ##tracks = get_tracks(tracks_f)
    #print(len(tracks))
    #tracks = filter_tracks(tracks, spots)
    #print(len(tracks))
    ##draw_tracks(tracks, spots, d, destination)

