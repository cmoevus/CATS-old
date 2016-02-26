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
from networkx.algorithms.components.connected import connected_components
from colorsys import hls_to_rgb
from random import randrange

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

# Blob detection
def find_blobs(img, blur=1, threshold=0.001):
    return blob_log(ndimage.gaussian_filter(img, blur), threshold=threshold, min_sigma=1)

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

# Import spots
def get_spots(source):
    spots = list()
    with open(source, 'r') as blobs:
        for y, x, s, t in [blob.split('\t') for blob in blobs]:
            spots.append((int(y), int(x), int(s), int(t)))
    return sorted(spots, key=lambda b: b[3])

# Make tracks out of continuous blobs over time
def make_tracks(spots, i_folder, destination):
    # Variables
    max_disp = [3, 3] # Maximum displacement, in pixels
    max_blink = 10 # Maximum blinking, in frames
    min_length = 3 # Minimum number of data points in a track

    # Reorganize spots by frame
    spots = np.array(spots, dtype='i8')
    n_frames = spots[:, 3].max() + 1
    frames = [[] for f in range(n_frames)]
    i = 0
    for spot in spots:
        frames[spot[3]].append(list(spot[:2])+[i])
        i += 1

    # Make optimal pairs
    a_pairs = list()
    for f in range(n_frames-1):

        # Measure the distances between spots
        d = np.abs(np.array(frames[f])[:, np.newaxis,:2] - np.array(frames[f+1])[:,:2])

        # Find the optimal pairs
        x = np.equal(d, np.min(d.transpose(1, 0, 2), axis=1))
        y = np.equal(d.transpose(1, 0, 2), np.min(d, axis=1)).transpose(1, 0, 2)
        pairs = ma.array(d, mask=np.logical_not(np.logical_and(x, y)))

        # Filter out the distances that exceed max_disp
        pairs = ma.masked_where(pairs > max_disp, pairs)
        p_shape = pairs.shape
        pairs = ma.mask_rows(pairs.ravel().reshape(p_shape[0]*p_shape[1], p_shape[2])).reshape(*p_shape)

        # Organize in actual 'pairs' (though, really, double triplets)
        pairs = set(tuple(pair) for pair in np.array(np.where(pairs.mask == False)).T[:,:2])
        for track, spot in pairs:
            a_pairs.append([frames[f][track][2], frames[f+1][spot][2]])

    # Merge pairs with common spots to form tracks. Using graph stuff
    G = nx.Graph()
    for p in a_pairs:
        G.add_nodes_from(p)
        G.add_edges_from([p])

    # Quality control for tracks
    tracks = list()
    for track in connected_components(G):
        if len(track) >= min_length:
            if np.abs(np.diff(sorted([spots[s][3] for s in track]))).max() <= max_blink:
                tracks.append(sorted(track, key=lambda a: spots[a][3]))

    # Draw tracks
    colors = [np.array(hls_to_rgb(randrange(0, 360)/360, randrange(20, 100, 10)/100, randrange(20, 80, 10)/100))*255 for i in tracks]
    frames = [[] for f in range(n_frames)]
    i = 0
    for track in tracks:
        for s in track:
            s = spots[s]
            frames[s[3]].append(list(s[:2])+[i])
        i += 1

    # Write RGB images
    j = 0
    for f in sorted(os.listdir(i_folder)):

        # Make images RGB
        i = ndimage.imread(i_folder+'/'+f)
        shape = list(i.shape)
        ni = np.zeros([3] + shape)
        ni[...,:] = i/i.max()*255
        ni = ni.transpose(1, 2, 0)

        # Write spots
        for s in frames[j]:
            print(colors[s[2]])
            ni[s[0],s[1],:] = colors[s[2]]
        misc.imsave(os.path.normpath(destination)+'/'+str(j)+'.tif', ni)
        j += 1

    return tracks

# Output
if __name__ == '__main__':
    #img = cv2.imread('TestData/01. half mL per min0000.tif', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #img = ndimage.imread('TestData/01. half mL per min0000.tif')

    # Stuff
    f = '/home/corentin/HR3rd spots.txt'
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
    spots = get_spots(f)
    t = time()
    tracks = make_tracks(spots, d, destination)
    print(time()-t)