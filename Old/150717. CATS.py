#!env /usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2, os
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
from scipy import ndimage, misc
from skimage.feature import blob_log, blob_dog, peak_local_max
from skimage.filters import threshold_otsu, gaussian_filter
from skimage.exposure import equalize_hist
from time import time
import networkx as nx
from colorsys import hls_to_rgb
from random import randrange
from math import floor, ceil

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
def find_barriers(img):
    # Invert and clean the image
    img = 255 - cv2.imread(img, 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.dilate(img, np.ones((5, 1), np.uint8), iterations=1)
    img = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    # Find lines
    lines = cv2.HoughLines(img, 1, 1, int(img.shape[0]/3))
    barriers = list()
    try:
        for rho, theta in lines[0]:
            if theta < 0.1:
                barriers.append(rho)
    except:
        pass
    return barriers

# Blob detection
def find_blobs(img, blur=1, threshold=0.001):
    return blob_log(ndimage.gaussian_filter(img, blur), threshold=threshold, min_sigma=1)

# Writes spots to a file
def write_spots(spots, f):
    with open(f, 'w') as record:
        for spot in spots:
            record.write('\t'.join([str(x) for x in spot])+'\t{0}'.format(i)+'\n')

# Import spots
def get_spots(source, xlims = None, ylims = None):
    spots = list()
    with open(source, 'r') as blobs:
        if xlims != None and ylims != None:
            for y, x, s, t in [blob.split('\t') for blob in blobs]:
                y, x, s, t = int(y), int(x), int(s), int(t)
                if xlims[0] <= x <= xlims[1] and ylims[0] <= y <= ylims[1]:
                    spots.append((y, x, s, t))
        else:
            for y, x, s, t in [blob.split('\t') for blob in blobs]:
                spots.append((int(y), int(x), int(s), int(t)))

    return sorted(spots, key=lambda b: b[3])

# Make tracks out of continuous blobs over time
def make_tracks(spots):
    # Variables
    max_disp = [3, 3] # Maximum displacement, in pixels
    max_blink = 20 # Maximum blinking, in frames
    min_length = 5 # Minimum number of data points in a track

    # Reorganize spots by frame
    n_frames = np.array(spots, dtype='i8')[:, 3].max() + 1
    frames = [[] for f in range(n_frames)]
    i = 0
    for spot in spots:
        frames[spot[3]].append(list(spot[:2])+[i])
        i += 1

    # Make optimal pairs for all acceptable frame intervals (as defined in max_blink
    a_pairs = list()
    for delta in range(1, max_blink+1):

        print('Delta frames: {0}'.format(delta))
        for f in range(n_frames-delta):

            # Measure the distances between spots
            d = np.abs(np.array(frames[f])[:, np.newaxis,:2] - np.array(frames[f+delta])[:,:2])

            # Find the optimal pairs
            x = np.equal(d, np.min(d.transpose(1, 0, 2), axis=1))
            y = np.equal(d.transpose(1, 0, 2), np.min(d, axis=1)).transpose(1, 0, 2)
            pairs = ma.array(d, mask=np.logical_not(np.logical_and(x, y)))

            # Filter out the distances that exceed max_disp
            pairs = ma.masked_where(pairs > max_disp, pairs)
            p_shape = pairs.shape
            pairs = ma.mask_rows(pairs.ravel().reshape(p_shape[0]*p_shape[1], p_shape[2])).reshape(*p_shape)

            # Organize in pairs, or edges, for graph purposes
            pairs = set(tuple(pair) for pair in np.array(np.where(pairs.mask == False)).T[:,:2])
            for track, spot in pairs:
                a_pairs.append([frames[f][track][2], frames[f+delta][spot][2]])

    # Using graph stuff
    G = nx.Graph()
    nodes = set()
    [nodes.update((s1, s2)) for s1, s2 in a_pairs]
    G.add_nodes_from(nodes)
    for p in a_pairs:
        G.add_edges_from([p])

    # Quality control for tracks
    tracks = list()
    for track in nx.connected_components(G):
        if len(track) >= min_length:
            if np.abs(np.diff(sorted([spots[s][3] for s in track]))).mean() <= max_blink/3 if max_blink/3 > 1 else 1:
                tracks.append(sorted(track, key=lambda a: spots[a][3]))

    return tracks

# Writes tracks to a file, to be reused with the same spots file and the same limits.
def write_tracks(tracks, f):
    with open(f, 'w') as record:
        for track in tracks:
            record.write(' '.join([str(x) for x in track])+'\n')

# Import spots
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
def draw_tracks(tracks, spots, d, destination):
    n_frames = np.array(spots)[:, 3].max() + 1

    # Make colors
    colors = [np.array(hls_to_rgb(randrange(0, 360)/360, randrange(20, 100, 10)/100, randrange(20, 80, 10)/100))*255 for i in tracks]

    # Separate tracks per image, because memory is short
    frames = [[] for f in range(n_frames)]
    i = 0
    for track in tracks:
        for s in track:
            s = spots[s]
            frames[s[3]].append(list(s[:2])+[i])
        i += 1

    # Write tracks RGB images
    j = 0
    for f in sorted(os.listdir(d)):

        # Make images RGB
        i = ndimage.imread(d+'/'+f)
        shape = list(i.shape)
        ni = np.zeros([3] + shape)
        ni[...,:] = i/i.max()*255
        ni = ni.transpose(1, 2, 0)

        # Write spots
        for s in frames[j]:
            ni[s[0],s[1],:] = colors[s[2]]
        misc.imsave(os.path.normpath(destination)+'/'+str(j)+'.tif', ni)
        j += 1

# Draws a SY Plot
def sy_plot(tracks, spots, binsize=3):
    '''
    Draws a SY (Stacked Ys) plot based on the tracks
    '''

    # Define the limits in X
    lims = (0, max([spots[spot][1] for track in tracks for spot in track]))

    # Create a dic of position -> lengths of tracks
    bins = np.arange(0, lims[1], binsize)
    y = dict()
    for i in range(0, len(bins)):
        y[i] = list()

    n = 0
    for track in tracks:
        frames = [spots[spot][3] for spot in track]
        length = max(frames) - min(frames) + 1
        position = np.mean([spots[spot][1] for spot in track])
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


def histogram(binsize=3, tracks=False):
    '''
    Return an histogram of the position of the spots in x.
    '''
    lims = self.get_xlims()
    bins = (lims[1]  - lims[0])/binsize
    if tracks == False:
        data = [spot[1] for spot in self.spots]
    else:
        data = [np.mean([self.spots[spot][1] for spot in track.values()]) for track in self.tracks]

    plt.figure()
    plt.xlabel('Position on DNA (pixels)')
    plt.ylabel('Number of interactions')
    plt.hist(data, bins=bins)
    plt.savefig('histogram.png', color='g')
    plt.close()

# Output
if __name__ == '__main__':
    #img = cv2.imread('TestData/01. half mL per min0000.tif', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #img = ndimage.imread('TestData/01. half mL per min0000.tif')

    # Stuff
    spots_f = '/home/corentin/HR3rd.spots'
    tracks_f = '/home/corentin/HR3rd.tracks'
    d = '/home/corentin/HR nucleosome 565 3rd barrier/'
    destination = '/home/corentin/HR nucleosome 565 3rd barrier results/'

    #
    # A. FIND SPOTS
    #
    #gaussian_blur = 0.5
    #threshold = 0.001
    #cutoff = 0

    #record = open(f, 'w')
    #i, spots, images = 0, list(), list()
    #for f in sorted(os.listdir(d)):

        #if i >= cutoff:
            ## Find spots
            #img = ndimage.imread(d+'/'+f)
            #b = find_blobs(img, blur=gaussian_blur, threshold=threshold)
            #print('Frame {0} done, {1} spots\r'.format(i, len(b)))

            ## Write the images down
            #shape = list(img.shape)
            #shape.append(3)
            #print(shape)
            #rgb = np.zeros(shape, dtype='int')
            #im = ndimage.gaussian_filter(img, gaussian_blur)/img.max()*255
            #rgb[...,0] = im
            #rgb[...,1] = im
            #rgb[...,2] = im
            ## Record the spot
            #for s in b:
                #record.write('\t'.join([str(x) for x in s])+'\t{0}'.format(i)+'\n')
                #y = int(round(s[0],0)) if round(s[0],0) < shape[0] - 1 else shape[0] - 1
                #x = int(round(s[1],0)) if round(s[1],0) < shape[1] - 1 else shape[1] - 1
                #rgb[y:y+1,x:x+1] = (255, 0, 0)
            #misc.imsave(os.path.normpath(destination)+'/'+str(i)+'.tif', rgb)
        #i += 1

    #
    # B. Form tracks
    #
    spots = get_spots(spots_f, (2, 45), (5, 507))
    #t = time()
    #tracks = make_tracks(spots)
    #print(time()-t)
    #print(len(tracks))
    #write_tracks(tracks, tracks_f)
    #draw_tracks(tracks, spots, d, destination)

    #tracks = make_tracks(spots)
    tracks = get_tracks(tracks_f)

    sy_plot(tracks, spots, 2)
    ## Dwell time vs binding time
    #fps = 1/0.06
    #binsize = int(ceil(1000/fps))
    #exp_len = int(ceil(10000/fps))
    #bins = [[] for t in range(0, exp_len, binsize)]
    #for track in tracks:
        #frames = [spots[s][3] for s in track]
        #if len(frames) < 9500:
            #m = min(frames)
            #dt = (max(frames) - m)/fps
            #bins[int(floor((m/fps)/binsize))].append(dt)

    ## All the dwell times
    #xs = list()
    #ys = list()
    #for i, b in enumerate(bins):
        #xs.extend([i*binsize for d in b])
        #ys.extend(b)
    #plt.plot(xs, ys, 'g.', label='Calculated dwell times')

    ## Mean dwell tims
    #y = [np.mean(b) for b in bins]
    #x = range(0, exp_len, binsize)
    #plt.plot(x, y, 'ro', label='Mean dwell time: {0:.2f}s'.format(np.mean(ys)))
    ##plt.xlim(-1, 11)
    #plt.ylabel('Mean dwell time (s)')
    #plt.yscale('log')
    #plt.xlabel('Binned time of initial binding (s)')
    #plt.legend()
    #plt.savefig('150718. Dwell time vs initial binding time.png')

    # Spreads
    #dts = list()
    #xs, ys = list(), list()
    #for track in tracks:
        #track = sorted(track, key= lambda a: spots[a][3])
        #coords = np.array([spots[s][:2] for s in track])
        #xs.extend(np.diff(np.array(coords)[:,1]))
        #ys.extend(np.diff(np.array(coords)[:,0]))

        ## Dwell times
        ##frames = sorted([spots[s][3] for s in track])
        ##dt = frames[-1]-frames[0]
        ##dts.append(dt)
    #f, (ax1, ax2) = plt.subplots(2)
    #ax1.hist(xs, bins=50)
    #ax2.hist(ys, bins=50)
    #plt.show()