# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function

from sys import stdout
from time import time
from multiprocessing import Pool, cpu_count
import numpy as np
import scipy as sp
import scipy.linalg as sl
from numpy import ma
from skimage.feature import blob_log
from scipy import ndimage
from scipy.optimize import curve_fit
from math import floor, ceil
import networkx as nx
from matplotlib import pyplot as plt
from ..content.particles import Particles, Particle
from .noise_simple import fit_noise
from ..utils import graph

__all__ = ['find_stationary_particles', 'detect_spots', 'find_blobs', 'link_spots', 'gaussian_2d']


def find_stationary_particles(source, blur, threshold, keep_unfit=False, max_blink=25, max_disp=(2, 2), max_processes=None, verbose=True):
    """
    Find the location of particles that do not move much in the given source.

    Arguments:
        blur: (float) see ndimage.gaussian_filter()'s 'sigma' argument
        threshold (float): see scipy.feature.blob_log()'s 'threshold' argument
        max_blink: (int) max number of frames a particle can 'disappear' before it is considered a different particle
        max_disp: (2-tuple) max distance (x, y) the particle can move between two frames before it is considered a different particle.
        keep_unfit: (bool) if spr is True, set to True to keep the spots that could not be fitted with a 2D Gaussian function (potentially lesser quality detections).
        max_processes: (int) the maximum number of simultaneous processes. Max and default value is the number of CPUs in the computer.
        verbose: (bool) give information while processing

    Return a Particles object with each Particle within containing the minimal fields and:
        s: (float) sigma of the Gaussian of the detection
        a: (float) amplitude of the Gaussian of the detection (intensity at peak)

    """
    spots = detect_spots(source, blur, threshold, keep_unfit, max_processes, verbose)
    tracks = link_spots(spots, max_blink, max_disp, verbose)
    particles = Particles(processor=find_stationary_particles, blur=blur, threshold=threshold, keep_unfit=keep_unfit, max_blink=max_blink, max_disp=max_disp, max_processes=max_processes, verbose=True, sources=source)
    for track in tracks:
        t = list()
        for s in track:
            t.append(spots[s])
        particle = np.array(t, dtype=t[0].dtype).view(Particle)
        particle.source = source
        particles.append(particle)
    return particles


def detect_spots(source, blur, threshold, keep_unfit=False, max_processes=None, verbose=True):
    """
    Find the gaussians in the given images.

    Arguments:

    blur: see ndimage.gaussian_filter()'s 'sigma' argument
    threshold: see scipy.feature.blob_log()'s 'threshold' argument
    keep_unfit: (bool) if spr is True, set to True to keep the spots that could not be fitted with a 2D Gaussian function (potentially lesser quality detections).
    max_processes: the maximum number of simultaneous processes. Max value should be the number of CPUs in the computer.
    verbose: Writes about the time and frames and stuff as it works
    """
    max_processes = cpu_count() if max_processes is None else max_processes
    # Multiprocess through it
    t = time()
    spots, pool = list(), Pool(max_processes)
    # for blobs in iter(find_blobs(i, blur, threshold, keep_unfit, j) for j, i in enumerate(source.read())):
    for blobs in pool.imap(find_blobs, iter((i, blur, threshold, keep_unfit, j) for j, i in enumerate(source.read()))):
        if verbose == True:
            print('\rFound {0} spots in frame {1}. Process started {2:.2f}s ago.         '.format(len(blobs), blobs[0][-1] if len(blobs) > 0 else 'i', time() - t), end='')
            stdout.flush()
        spots.extend(blobs)
    pool.close()
    pool.join()

    if verbose is True:
        print('\rFound {0} spots in {1} frames in {2:.2f}s{3}'.format(len(spots), source.length, time() - t, " " * 30), end='\n')

    return np.array(spots, dtype={'names': ('x', 'y', 'sx', 'sy', 'i', 't'), 'formats': (float, float, float, float, int, int)}).view(np.recarray)


def find_blobs(*args):
    """
    Find blobs in an image. Return a list of spots as (y, x, s, i).

    List of spots:
    y and x are the locations of the center, s is the standard deviation of the gaussian kernel and i is the intensity at the center.

    Arguments:  (as a list, for multiprocessing)
        image: a numpy array representing the image to analyze
        blur: see ndimage.gaussian_filter()'s 'sigma' argument
        threshold: see scipy.feature.blob_log()'s 'threshold' argument
        keep_unfit: (bool) if spr is True, set to True to keep the spots that could not be fitted with a 2D Gaussian function (potentially lesser quality detections).
        extra: information to be added at the end of the blob's properties
    """
    args = args[0] if len(args) == 1 else args
    image, blur, threshold, keep_unfit, frame = args[0], args[1], args[2], args[3], args[4]
    b = blob_log(ndimage.gaussian_filter(image, blur), threshold=threshold, min_sigma=1)
    blobs = list()
    for y, x, s in b:
        blobs.append((x, y, s, image[y][x], frame))
    return fit_blobs_moments(image, blobs, keep_unfit)


def fit_blobs_moments(img, blobs, keep_unfit):
    """
    Find the center of mass of the blob.

    following indications from:
    http://scitation.aip.org/content/aip/journal/rsi/78/5/10.1063/1.2735920
    """
    spr_blobs = list()
    for blob in blobs:

        # A. Get an ROI of the image
        r = max(blob[2] + 1 * 2, 3)
        ylim, xlim = img.shape
        slice_y = slice(max(0, int(floor(blob[1] - r))), min(int(ceil(blob[1] + r)) + 1, ylim))
        slice_x = slice(max(0, int(floor(blob[0] - r))), min(int(ceil(blob[0] + r)) + 1, xlim))
        x_values, y_values = img[slice_y, blob[0]], img[blob[1], slice_x]
        x_min, y_min = x_values.min(), y_values.min()

        # B. Find the center of mass
        x = np.sum([(p + 0.5) * i for p, i in enumerate(x_values)]) / np.sum(x_values)
        y = np.sum([(p + 0.5) * i for p, i in enumerate(y_values)]) / np.sum(y_values)

        # C. Get the parameters
        x0, y0 = x + slice_x.start, y + slice_y.start
        A = img[min(int(y0), ylim - 1), min(int(x0), xlim - 1)]
        sx = np.std([j for i, v in enumerate(x_values) for j in [i] * (v - x_min)])
        sy = np.std([j for i, v in enumerate(y_values) for j in [i] * (v - y_min)])
        spr_blobs.append((x0, y0, sx, sy, A, blob[4]))

        # # Show fit
        # print((round(x, 2), round(y, 2), round(sx, 2), round(sy, 2), A, blob[3]))
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.bar(range(0, len(x_values)), x_values, width=1)
        # xf = np.linspace(0, len(x_values), len(x_values) * 10)
        # plt.plot(xf, gaussian_noise(xf, A, x, sx, noise), 'r')
        # plt.subplot(1, 2, 2)
        # plt.bar(range(0, len(y_values)), y_values, width=1)
        # yf = np.linspace(0, len(y_values), len(y_values) * 10)
        # plt.plot(yf, gaussian_noise(yf, A, y, sy, noise), 'r')
        # plt.show()

    return spr_blobs


def fit_blobs_curvefit_1D(img, blobs, keep_unfit):
    """
    Fit 2D Gaussians by splitting them in two 1D gaussians.
    """
    # A. Get the noise value, to provide an estimated amplitude for unfit particles
    if keep_unfit == True:
        try:
            fit, cov = fit_noise(img)
            noise = fit[1]
        except RuntimeError:
            noise = img.mean()

    # B. Try to fit 2 times 1D gaussians on given blobs (more precise than 1 time 2D Gaussian)
    spr_blobs = list()
    for blob in blobs:

        # B1. Get an ROI of the image
        r = max(blob[2] + 1 * 2, 3)
        ylim, xlim = img.shape
        slice_y = slice(max(0, int(floor(blob[1] - r))), min(int(ceil(blob[1] + r)) + 1, ylim))
        slice_x = slice(max(0, int(floor(blob[0] - r))), min(int(ceil(blob[0] + r)) + 1, xlim))
        x0, y0 = blob[0] - slice_x.start, blob[1] - slice_y.start
        x_values, y_values = img[slice_y, blob[0]], img[blob[1], slice_x]

        # B2. Fit the blob on the ROI
        try:
            fit_x, pcov_x = curve_fit(gaussian, np.arange(0.5, len(x_values) + 0.5), x_values, p0=(blob[3], x0, blob[2], x_values.min()))
            fit_y, pcov_y = curve_fit(gaussian, np.arange(0.5, len(y_values) + 0.5), y_values, p0=(blob[3], y0, blob[2], y_values.min()))
        except RuntimeError:
            if keep_unfit == True:
                spr_blobs.append((blob[0], blob[1], blob[2], blob[2], blob[3] - noise, blob[3], blob[4]))
            continue

        # B3. Extract the data out of the ROI and save it.
        cov_x, cov_y = np.diag(pcov_x), np.diag(pcov_y)
        A = sorted([(cov_x[0], fit_x[0]), (cov_y[0], fit_y[0])])[0][1]
        x0, y0, sx, sy = fit_x[1] + slice_x.start, fit_y[1] + slice_y.start, abs(fit_x[2]), abs(fit_y[2])
        spr_blobs.append((x0, y0, sx, sy, A, blob[3], blob[4]))

        # # Show fit
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.bar(range(0, len(x_values)), x_values, width=1)
        # xf = np.linspace(0, len(x_values), len(x_values) * 10)
        # plt.plot(xf, gaussian(xf, *fit_x), 'r')
        # plt.subplot(1, 2, 2)
        # plt.bar(range(0, len(y_values)), y_values, width=1)
        # yf = np.linspace(0, len(y_values), len(y_values) * 10)
        # plt.plot(yf, gaussian(yf, *fit_y), 'r')
        # plt.show()

    return spr_blobs


def fit_blobs_linalg(img, blobs, keep_unfit):
    """
    Fit 2D Gaussians by splitting them in two 1D gaussians.

    Implementation of the methods described in
        Hongwei Guo, "A Simple Algorithm for Fitting a Gaussian Function"
        IEEE Signal Processing Magazine, September 2011, pp. 134--137
    Author of the implementation: Stefan van der Walt, 2011
    """
    # A. Get the noise value, to provide an estimated amplitude for unfit particles
    try:
        fit, cov = fit_noise(img)
        noise = fit[1]
    except RuntimeError:
        noise = img.mean()

    # B. Try to fit 2 times 1D gaussians on given blobs (more precise than 1 time 2D Gaussian)
    spr_blobs = list()
    for blob in blobs:

        # B1. Get an ROI of the image
        r = max(blob[2] + 1 * 2, 4)
        ylim, xlim = img.shape
        slice_y = slice(max(0, int(floor(blob[1] - r))), min(int(ceil(blob[1] + r)) + 1, ylim))
        slice_x = slice(max(0, int(floor(blob[0] - r))), min(int(ceil(blob[0] + r)) + 1, xlim))
        x0, y0 = blob[0] - slice_x.start, blob[1] - slice_y.start
        x_values, y_values = img[slice_y, blob[0]] - noise, img[blob[1], slice_x] - noise

        # B2. Fit the blob on the ROI
        try:
            fit_x = fit_iterative(np.arange(0.5, len(x_values) + 0.5), x_values, N=10)
            fit_y = fit_iterative(np.arange(0.5, len(y_values) + 0.5), y_values, N=10)
        except RuntimeError:
            if keep_unfit == True:
                spr_blobs.append((blob[0], blob[1], blob[2], blob[2], blob[3] - noise, blob[3], blob[4]))
            continue

        # B3. Extract the data out of the ROI and save it.
        A = sorted([((fit_x[0] - blob[3])**2 / blob[3], fit_x[0]), ((fit_y[0] - blob[3])**2 / blob[3], fit_y[0])])[0][1]
        x0, y0, sx, sy = fit_x[1] + slice_x.start, fit_y[1] + slice_y.start, abs(fit_x[2]), abs(fit_y[2])
        spr_blobs.append((x0, y0, sx, sy, A, blob[3], blob[4]))

        # # Show fit
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.bar(range(0, len(x_values)), x_values, width=1)
        # xf = np.linspace(0, len(x_values), len(x_values) * 10)
        # plt.plot(xf, gaussian(xf, *fit_x), 'r')
        # plt.subplot(1, 2, 2)
        # plt.bar(range(0, len(y_values)), y_values, width=1)
        # yf = np.linspace(0, len(y_values), len(y_values) * 10)
        # plt.plot(yf, gaussian(yf, *fit_y), 'r')
        # plt.show()

    return spr_blobs


def fit_direct(x, y, F=0, weighted=True, _weights=None):
    """Fit a Gaussian to the given data.

    Returns a fit so that y ~ gauss(x, A, mu, sigma)

    Parameters
    ----------
    x : ndarray
        Sampling positions.
    y : ndarray
        Sampled values.
    F : float
        Ignore values of y &lt;= F.
    weighted : bool
        Whether to use weighted least squares.  If True, weigh
        the error function by y, ensuring that small values
        has less influence on the outcome.

    Additional Parameters
    ---------------------
    _weights : ndarray
        Weights used in weighted least squares.  For internal use
        by fit_iterative.

    Returns
    -------
    A : float
        Amplitude.
    mu : float
        Mean.
    std : float
        Standard deviation.

    """
    mask = (y > F)
    x = x[mask]
    y = y[mask]

    if _weights is None:
        _weights = y
    else:
        _weights = _weights[mask]

    # We do not want to risk working with negative values
    np.clip(y, 1e-10, np.inf, y)

    e = np.ones(len(x))
    if weighted:
        e = e * (_weights**2)

    v = (np.sum(np.vander(x, 5) * e[:, None], axis=0))[::-1]
    A = v[sl.hankel([0, 1, 2], [2, 3, 4])]

    ly = e * np.log(y)
    ls = np.sum(ly)
    x_ls = np.sum(ly * x)
    xx_ls = np.sum(ly * x**2)
    B = np.array([ls, x_ls, xx_ls])

    (a, b, c), res, rank, s = np.linalg.lstsq(A, B)

    A = np.exp(a - (b**2 / (4 * c)))
    mu = -b / (2 * c)
    sigma = sp.sqrt(-1 / (2 * c))

    return A, mu, sigma


def fit_iterative(x, y, F=0, weighted=True, N=10):
    """Fit a Gaussian to the given data.

    Returns a fit so that y ~ gauss(x, A, mu, sigma)

    This function iteratively fits using fit_direct.

    Parameters
    ----------
    x : ndarray
        Sampling positions.
    y : ndarray
        Sampled values.
    F : float
        Ignore values of y &lt;= F.
    weighted : bool
        Whether to use weighted least squares.  If True, weigh
        the error function by y, ensuring that small values
        has less influence on the outcome.
    N : int
        Number of iterations.

    Returns
    -------
    A : float
        Amplitude.
    mu : float
        Mean.
    std : float
        Standard deviation.

    """
    y_ = y
    for i in range(N):
        p = fit_direct(x, y, weighted=True, _weights=y_)
        A, mu, sigma = p
        y_ = gaussian(x, A, mu, sigma)

    return np.real(A), np.real(mu), np.real(sigma)


def fit_blobs_curvefit_2D(img, blobs, keep_unfit):
    """
    Fit a Gaussian curve on the blobs in an image. Return the given list of blobs with the fitted values.

    Arguments:
        img: the ndarray of the image containing the blobs
        blobs: a list of blobs (x, y, sigma, intensity, ...) that need to go subpixel resolution. All extra information after 'intensity'  will be kept in the output.
    """
    spr_blobs = list()
    for blob in blobs:
        r = blob[2] + 1 * np.sqrt(2)
        ylim, xlim = img.shape
        y = (max(0, int(floor(blob[1] - r))), min(int(ceil(blob[1] + r)) + 1, ylim))
        x = (max(0, int(floor(blob[0] - r))), min(int(ceil(blob[0] + r)) + 1, xlim))
        data = img[y[0]:y[1], x[0]:x[1]]
        coords = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
        try:
            fit, cov = curve_fit(gaussian_2d, coords, data.ravel(), p0=(blob[3], blob[0] - x[0], blob[1] - y[0], blob[2], data.min()))
            spr_blob = [x[0] + fit[1], y[0] + fit[2], abs(fit[3]), fit[0]]
            if len(blob) > 4:
                spr_blob.extend(blob[4:])
            if x[0] <= spr_blob[0] <= x[1] and y[0] <= spr_blob[1] <= y[1]:
                spr_blobs.append(tuple(spr_blob))
            else:
                raise ValueError
        except:
            if keep_unfit == True:
                spr_blobs.append(blob)

    return spr_blobs


def link_spots_simple(spots, max_blink, max_disp, verbose):
    """Correlate spots through time as tracks (or particles)."""
    # Reorganize spots by frame and prepare the Graph
    G = nx.DiGraph()
    n_frames = max(spots['t']) + 1
    frames = [[] for f in range(n_frames)]
    for i, spot in enumerate(spots):
        frames[spot['t']].append((spot['x'], spot['y'], i))
        G.add_node(i, frame=spot['t'])

    # Find optimal pairs for all acceptable frame intervals (as defined in max_blink)
    for delta in range(1, max_blink + 1):
        if verbose is True:
            print('\rDelta frames: {0}'.format(delta), end='')
            stdout.flush()
        for f in range(n_frames - delta):
            if len(frames[f]) == 0 or len(frames[f + delta]) == 0:
                continue
            # Matrix of distances between spots
            d = np.abs(np.array(frames[f])[:, np.newaxis, :2] - np.array(frames[f + delta])[:, :2])

            # Filter out the spots with distances that excess max_disp in x and/or y
            disp_filter = d - max_disp >= 0
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
    tracks = list()
    for track in nx.weakly_connected_component_subgraphs(G):
        t_frames = [spots[s]['t'] for s in track]

        # Good tracks
        if len(t_frames) == len(set(t_frames)):
            tracks.append(sorted(track.nodes(), key=lambda s: spots[s]['t']))

        # Ambiguous tracks
        # This is a work in progress, if progress there ever is
        else:
            nodes = track.nodes()
            track = dict([(spots[s]['t'], []) for s in nodes])
            for s in nodes:
                track[spots[s]['t']].append(s)
            ts = sorted(track.keys())
            for t in ts[:-1]:
                if len(track[t]) > 1:
                    now = track[t]
                    t_after = ts.index(t) + 1
                    after = track[ts[t_after]]
                    scores = np.abs(np.array([[spots[s]['x'], spots[s]['y']] for s in now])[:, np.newaxis] - np.array([[spots[s]['x'], spots[s]['y']] for s in after]))
                    scores = scores[:, :, 0] + scores[:, :, 1]
                    pair = np.where(scores == scores.min())
                    if len(pair[0]) > 1:
                        pair = (np.array(pair).T)[0]
                    now, after = [now[pair[0]]], [after[pair[1]]]
            track = sorted([t[0] for t in track.values()], key=lambda a: spots[a]['t'])
            ts = [spots[s]['t'] for s in track]
            tracks.append(track)

    if verbose is True:
        print('\nFound {0} particles'.format(len(tracks)))

    return tracks


def link_spots(spots, max_blink, max_disp, verbose):
    """Correlate spots through time as tracks (or particles)."""
    start_time = time()
    # Reorganize spots by frame and prepare the Graph
    G = nx.DiGraph()
    n_frames = max(spots['t']) + 1
    frames = [[] for f in range(n_frames)]
    for i, spot in enumerate(spots):
        frames[spot['t']].append((spot['x'], spot['y'], i))
        G.add_node(i, frame=spot['t'])

    # Find optimal pairs for all acceptable frame intervals (as defined in max_blink)
    for delta in range(1, max_blink + 1):
        if verbose is True:
            print('\rDelta frames: {0}'.format(delta), end='')
            stdout.flush()
        for f in range(n_frames - delta):
            if len(frames[f]) == 0 or len(frames[f + delta]) == 0:
                continue
            # Matrix of distances between spots
            d = np.abs(np.array(frames[f])[:, np.newaxis, :2] - np.array(frames[f + delta])[:, :2])

            # Filter out the spots with distances that excess max_disp in x and/or y
            disp_filter = d - max_disp >= 0
            disp_mask = np.logical_or(disp_filter[:, :, 0], disp_filter[:, :, 1])

            # Reduce to one dimension
            d = np.sqrt(d[:, :, 0]**2 + d[:, :, 1]**2)

            # Find the optimal pairs
            f_best = d != np.min(d, axis=0)
            fd_best = (d.T != np.min(d, axis=1)).T
            mask = np.logical_and(f_best, fd_best)
            mask = np.logical_or(mask, disp_mask)
            pairs = ma.array(d, mask=mask)

            # Organize in pairs, or edges, for graph purposes
            for s1, s2 in np.array(np.where(pairs.mask == False)).T:
                G.add_edge(frames[f][s1][2], frames[f + delta][s2][2],
                           weight=d[s1][s2])

    # Resolve ambiguities i. e. nodes with more than one incoming or outgoing edge
    if verbose is True:
        print('\rResolving ambiguities...', end='')
        stdout.flush()
    tracks = list()
    for track in nx.weakly_connected_component_subgraphs(G):
        track_sort = sorted([s for s in track.nodes()], key=lambda a: spots[a]['t'])
        track_reduced = graph.transitive_reduction(track, order=track_sort)
        for n, i in track_reduced.in_degree().items():
            if i > 1:
                keep_nearest_neighbor_only(track, track_reduced, n, track_reduced.predecessors(n))
        for n, i in track_reduced.out_degree().items():
            if i > 1:
                keep_nearest_neighbor_only(track, track_reduced, n, track_reduced.successors(n))

        for t in nx.weakly_connected_components(track_reduced):
            tracks.append(sorted(t, key=lambda s: spots[s]['t']))
        # Draw the graph
        # n_per_frame = dict()
        # for s in track.nodes():
        #     f = spots[s]['t']
        #     if f not in n_per_frame:
        #         n_per_frame[f] = []
        #     n_per_frame[f].append(s)
        # plt.figure()
        # plt.xlabel('Frame')
        # nodes = dict()
        # for f, ss in n_per_frame.items():
        #     for y, s in enumerate(ss):
        #         plt.text(s, f, y + 1)
        #         nodes[s] = (f, y  + 1)
        # for e in track_reduced.edges():
        #     n0, n1 = nodes[e[0]], nodes[e[1]]
        #     plt.plot([n0[0], n1[0]], [n0[1], n1[1]])
        # plt.show()

    if verbose is True:
        print('\rFound {0} particles in {1:.2f}s.{2}'.format(len(tracks), time() - start_time, " " * 15), end='\n')

    return tracks


def keep_nearest_neighbor_only(track, track_reduced, n, a):
    """
    Delete the edges from a DiGraph that are not the 'best' neighbors.

    Arguments:
        track: (nx.DiGraph) the track containing the nodes
        track_reduced: (nx.DiGraph)
        n: the main node
        a: list of competing nodes
    """
    neighbors = set(track.predecessors(n) + track.successors(n))
    common_neighbors = sorted([(len(neighbors.intersection(track.predecessors(i) + track.successors(i))), i) for i in a])
    for l, a_n in common_neighbors[:-1]:
        track_reduced.remove_edge(*sorted([n, a_n]))
        track.remove_edge(*sorted([n, a_n]))


def gaussian_2d(coords, A, x0, y0, s, B):
    """
    Draw a 2D gaussian with given properties.

    Arguments:
        A: Amplitude of the Gaussian
        x0: position of the center, in x
        y0: position of the center, in y
        s: standard deviation
        B: background noise
    """
    x, y = coords
    if A < B:
        return np.zeros(coords.shape)
    return (A * np.exp(((x - x0)**2 + (y - y0)**2) / (-2 * s**2)) + B).ravel()


def gaussian_noise(x, A, m, s, n):
    """
    Return a Gaussian.

    Arguments:
        x: the values in x
        A: the amplitude
        m: the mean
        s: the standard deviation
        n: the baseline noise
    """
    return (A - n) * np.e**((-(x - m)**2) / (2 * s**2)) + n


def fixed_gaussian(A, m):
    """Return a Gaussian function with all parameters but std fixed."""
    def fg(x, s, n):
        return A * np.e**((-(x - m)**2) / (2 * s**2)) + n
    return fg


def gaussian(x, A, m, s):
    """
    Return a Gaussian.

    Arguments:
        x: the values in x
        A: the amplitude
        m: the mean
        s: the standard deviation
        n: the baseline noise
    """
    return A * np.e**((-(x - m)**2) / (2 * s**2))

__processor__ = {
    'particles': {'stationary': find_stationary_particles},
    'gaussians': {'log_and_fit': detect_spots}
}
