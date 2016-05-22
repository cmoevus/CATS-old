# -*- coding: utf8 -*-
"""Find steps in the given data."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp


def find_steps(data, alpha=0.05):
    """
    Find steps in the given data.

    This function considers that each plateau corresponds to a normal distribution. It will select steps in the data until the difference between two plateaus, as estimated by a t-test, are not significant (p-value > alpha).

    data <list>: datapoints
    alpha <float>: significance level to achieve
    """
    split_list, steps = [data], list()
    # Find plateaus that are statistically significant
    while len(split_list) != 0:
        plateau, D = split_list.pop(), np.inf
        for i in range(2, len(plateau) - 2):
            d = lstsq(plateau, fit_steps([i], plateau))
            if d < D:
                D = d
                step = i
        plateaux = plateau[:step], plateau[step:]
        T, pval = sp.stats.ttest_ind(*plateaux, equal_var=False)
        if pval <= alpha and np.isnan(pval) == False:
            split_list.extend(plateaux)
            steps.append(step)

    # Check significance
    steps = sorted(steps)
    boundaries = [0] + steps + [len(data)]
    i = 0
    while i < len(boundaries) - 2:
        p1 = data[boundaries[i]:boundaries[i + 1]]
        p2 = data[boundaries[i + 1]:boundaries[i + 2]]
        T, pval = sp.stats.ttest_ind(p1, p2, equal_var=False)
        if pval > alpha or np.isnan(pval) == True:
            print("({0}, {1}) and ({1}, {2}): {3} -> {4}".format(boundaries[i], boundaries[i + 1], boundaries[i + 2], pval, 'No'))
            steps.pop(i)
            boundaries = [0] + steps + [len(data)]
            i -= 1 if i > 0 else 0
        else:
            print("({0}, {1}) and ({1}, {2}): {3} -> {4}".format(boundaries[i], boundaries[i + 1], boundaries[i + 2], pval, 'Yes'))
            i += 1

    return steps


def find_steps2(data, alpha=0.01):
    """
    Find steps in the given data.

    This function considers that each plateau corresponds to a normal distribution. It will select steps in the data until the difference between two plateaus, as estimated by a t-test, are not significant (p-value > alpha).

    data <list>: datapoints
    alpha <float>: significance level to achieve
    """
    split_list, steps = [data], list()
    # Find plateaus that are statistically significant
    while len(split_list) != 0:
        plateau, D = split_list.pop(), np.inf
        for i in range(2, len(plateau) - 2):
            d = lstsq(plateau, fit_steps([i], plateau))
            if d < D:
                D = d
                step = i
        plateaux = plateau[:step], plateau[step:]
        if good_to_split(*plateaux, alpha=alpha):
            split_list.extend(plateaux)
            steps.append(step)

    # Check significance
    steps = sorted(steps)
    boundaries = [0] + steps + [len(data)]
    i = 0
    while i < len(boundaries) - 2:
        p1 = data[boundaries[i]:boundaries[i + 1]]
        p2 = data[boundaries[i + 1]:boundaries[i + 2]]
        if good_to_split(p1, p2, alpha) == False:
            # print("({0}, {1}) and ({1}, {2}): {3} -> {4}".format(boundaries[i], boundaries[i + 1], boundaries[i + 2], pval, 'No'))
            steps.pop(i)
            boundaries = [0] + steps + [len(data)]
            i -= 1 if i > 0 else 0
        else:
            # print("({0}, {1}) and ({1}, {2}): {3} -> {4}".format(boundaries[i], boundaries[i + 1], boundaries[i + 2], pval, 'Yes'))
            i += 1

    return steps


def good_to_split(p1, p2, alpha):
    """
    Test if two plateaux are statistically and visually different.

    Statistic: t-test p-value <= alpha
    Visual: mean1 + stdev1 <= mean2 and mean2 + stdev2 >= mean1
    """
    # Stats
    T, pval = sp.stats.ttest_ind(p1, p2, equal_var=False)
    stats = pval <= alpha and np.isnan(pval) == False

    # Visual
    params = sorted([(np.mean(p1), np.std(p1)), (np.mean(p2), np.std(p2))])
    visual = params[0][0] + params[0][1] <= params[1][0] and params[1][0] - params[1][1] >= params[0][1]

    return stats and visual


def find_steps_lstsq2(data, alpha=0.05):
    """
    Find steps in the given data.

    This function considers that each plateau corresponds to a normal distribution. It will select steps in the data until the difference between two plateaus, as estimated by a t-test, are not significant (p-value > alpha).

    data <list>: datapoints
    alpha <float>: significance level to achieve
    """
    split_list, steps, S = [data], list(), list()
    while len(S) <= len(data) / 10:
        plateau, D = split_list.pop(), np.inf
        for i in range(1, len(plateau) - 1):
            d = lstsq(plateau, fit_steps([i], plateau))
            if d < D:
                D = d
                step = i
        steps.append(step)
        S.append(lstsq(plateau, fit_counter_steps([step], plateau)) / D)
        split_list.extend([plateau[step:], plateau[:step]])
    step_number = S.index(max(S)) + 1
    steps = steps[:step_number]
    # return steps, fit_steps(steps, data), S
    return steps


def find_steps_chisq2(data, alpha=0.05):
    """
    Find steps in the given data.

    This function considers that each plateau corresponds to a normal distribution. It will select steps in the data until the difference between two plateaus, as estimated by a t-test, are not significant (p-value > alpha).

    data <list>: datapoints
    alpha <float>: significance level to achieve
    """
    split_list, steps, S = [data], list(), list()
    while len(S) <= len(data) / 10:
        plateau, D = split_list.pop(), np.inf
        for i in range(1, len(plateau) - 1):
            d = chi_square(plateau, fit_steps([i], plateau))
            if d < D:
                D = d
                step = i
        steps.append(step)
        S.append(chi_square(plateau, fit_counter_steps([step], plateau)) / D)
        split_list.extend([plateau[step:], plateau[:step]])
    step_number = S.index(max(S)) + 1
    steps = steps[:step_number]
    # return steps, fit_steps(steps, data), S
    return steps


def find_steps_lstsq(data):
    """Find steps in the given data."""
    steps, S = list(), list()
    max_steps = int(len(data) / 10)
    for j in range(max_steps):
        X2, step = np.inf, -1
        for i in range(1, len(data) - 1):
            temp_steps = steps + [i]
            temp_X2 = lstsq(data, fit_steps(temp_steps, data))
            if temp_X2 < X2:
                X2 = temp_X2
                step = i
        if step != -1:
            steps.append(step)
            S.append(lstsq(data, fit_counter_steps(steps, data)) / X2)
        else:
            print('WHAT?!')
    step_number = S.index(max(S)) + 1
    steps = steps[:step_number]
    # return steps, fit_steps(steps, data), S
    return steps


def lstsq(i, j):
    """Return the least square distance between the two datasets."""
    if len(i) != len(j):
        raise ValueError('Datasets must be of the same length.')
    x = 0
    for k, l in zip(i, j):
        x += (k - l)**2
    return x


def find_steps_chisquare(data):
    """
    Find steps in the given data.

    Using the Chi-Square algo from Kerssemakers, J. W. J. et al. Assembly dynamics of microtubules at molecular resolution. Nature 442, 709–712 (2006).
    """
    steps, S = list(), list()
    max_steps = int(len(data) / 10)
    for j in range(max_steps):
        X2, step = np.inf, -1
        for i in range(1, len(data) - 1):
            temp_steps = steps + [i]
            temp_X2 = chi_square(data, fit_steps(temp_steps, data))
            if temp_X2 < X2:
                X2 = temp_X2
                step = i
        if step != -1:
            steps.append(step)
            S.append(chi_square(data, fit_counter_steps(steps, data)) / X2)
        else:
            print('WHAT?!')
    step_number = S.index(max(S)) + 1
    steps = steps[:step_number]
    # return steps, fit_steps(steps, data), S
    return steps


def chi_square(i, j):
    """Measure the Chi Square value between dataset i and expected values j."""
    if len(i) != len(j):
        raise ValueError('Datasets must be of the same length.')
    X2 = 0
    for k, l in zip(i, j):
        X2 += (k - l)**2 / abs(l)
    return X2


def fit_steps(steps, data):
    """Fit the steps to the data and return a distribution of size len(data) with the fitted plateaux and steps."""
    fit = list()
    boundaries = [0] + sorted(steps) + [len(data)]
    for i in range(len(boundaries) - 1):
        plateau = data[boundaries[i]:boundaries[i + 1]]
        mean = np.mean(plateau)
        fit.extend([mean for j in plateau])
    return fit


def fit_counter_steps(steps, data):
    """Fit the counter steps to the data and return a distribution of size len(data) with the fitted plateaux and counter steps."""
    fit = list()
    boundaries = [0] + sorted(steps) + [len(data)]
    counterboundaries = [0]
    for i in range(len(boundaries) - 2):
        counterboundaries.append(int((boundaries[i + 1] - boundaries[i]) / 2) + boundaries[i])
    counterboundaries.append(len(data))
    for i in range(len(counterboundaries) - 1):
        plateau = data[counterboundaries[i]:counterboundaries[i + 1]]
        mean = np.mean(plateau)
        fit.extend([mean for i in plateau])
    return fit


def find_steps_ttest(data, alpha=0.05, d=2):
    """
    Find steps in the given data.

    alpha <float>: significance level for comparing two potential plateaux
    d <int>: distance from time 0 and from last timepoint to start comparisons. Has to be >= 2.
    """
    steps, plateaux = list(), list()
    plateaux.append(data)
    while len(plateaux) > 0:
        plateau, lowest_pvalue, pos = plateaux.pop(), alpha, -1
        for i in range(d, len(plateau) - d):
            stat, pval = sp.stats.ttest_ind(plateau[:i], plateau[i:], equal_var=False)
            # stat, pval = sp.stats.ks_2samp(plateau[:i], plateau[i:])
            if pval < lowest_pvalue:
                lowest_pvalue = pval
                pos = i
        if pos != -1:
            steps.append(pos)
            plateaux.append(plateau[:pos])
            plateaux.append(plateau[pos:])
    steps = sorted(steps)
    boundaries = [0] + steps + [len(data)]

    # # Merge back things that actually are the same...
    # i = 0
    # while i < len(boundaries) - 2:
    #     dists = data[boundaries[i]:boundaries[i + 1]], data[boundaries[i + 1]:boundaries[i + 2]]
    #     stat, pval = sp.stats.ttest_ind(dists[0], dists[1], equal_var=False)
    #
    #     # Not statistically different: merge
    #     if pval > alpha:
    #         print('NOT', i)
    #         steps.pop(i + 1)
    #         boundaries = [0] + steps + [len(data)]
    #         i -= 1 if i != 0 else 0
    #
    #     # Statistically significant: Go to the next step
    #     else:
    #         print('DOES', i)
    #         i += 1

    # Get plateaux
    plateaux = list()
    for b in range(len(boundaries) - 1):
        plateaux.append(data[boundaries[b]:boundaries[b + 1]])

    return steps, plateaux
