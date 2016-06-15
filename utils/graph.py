# -*- coding: utf8 -*-
"""
Implement useful graph theory functions.

Functions:
----------

transitive_reduction: return the transitive reduction of a graph.

transitive_closure: return the transivite closure of a graph.

"""

from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp
import networkx as nx


def transitive_reduction(G, order=None, adj_matrix=False):
    """
    Return the transitive reduction of a given graph.

    Uses numpy arrays if it can, or SciPy Sparse Matrices (CSR) in case of MemoryError.

    Based on Aho, A. V., Garey, M. R., Ullman, J. D. The Transitive
    Reduction of a Directed Graph. SIAM Journal on Computing 1, 131–137
    (1972).

    Arguments:
        order: the order in which the vertices appear in time (to fit the second condition). If None, uses the same order as in the graph
        adj_matrix: returns an adjacency matrix if True, a Graph if False
    """
    try:
        return transitive_reduction_np(G, order, adj_matrix)
    except MemoryError:
        return transitive_reduction_sp(G, order, adj_matrix)


def transitive_closure(G, order=None, adj_matrix=False):
    """
    Return the transivite closure of a graph.

    Uses numpy arrays if it can, or SciPy Sparse Matrices (CSR) in case of MemoryError.

    This method assumes that your graph is:
        1. Directed acyclic
        2. Organized so that a node can only interact with a node
        positioned after it, in the adjacency matrix

    Arguments:
        order: the order in which the vertices appear in time (to fit the second condition)
        adj_matrix: returns an adjacency matrix if True, a Graph if False
    """
    try:
        return transitive_closure_np(G, order, adj_matrix)
    except MemoryError:
        return transitive_closure_sp(G, order, adj_matrix)


def transitive_reduction_sp(G, order=None, adj_matrix=False):
    """
    Return the transitive reduction of a given graph using Scipy's Sparse Matrices.

    Based on Aho, A. V., Garey, M. R., Ullman, J. D. The Transitive
    Reduction of a Directed Graph. SIAM Journal on Computing 1, 131–137
    (1972).

    Arguments:
        order: the order in which the vertices appear in time (to fit the second condition). If None, uses the same order as in the graph
        adj_matrix: returns an adjacency matrix if True, a Graph if False
    """
    # Transitive closure
    MT = transitive_closure_sp(G, order, True)

    # Reorganize the adjacency matrix
    M = nx.to_scipy_sparse_matrix(G, nodelist=order, weight=None, format='csr')

    # Adjacency matrices operations
    Mt = sp.sparse.csr_matrix(M - M.dot(MT))

    # Return in the proper format
    if adj_matrix is True:
        return (Mt == 1)
    else:
        Gt = G.copy()
        edges = sp.sparse.find(Mt != M)
        for k in range(len(edges[0])):
            i, j = edges[0][k], edges[1][k]
            try:
                Gt.remove_edge(order[i], order[j])
            except:
                pass
        return Gt


def transitive_closure_sp(G, order=None, adj_matrix=False):
    """
    Return the transivite closure of a graph using SciPy's Sparse Matrices.

    This method assumes that your graph is:
        1. Directed acyclic
        2. Organized so that a node can only interact with a node
        positioned after it, in the adjacency matrix

    Arguments:
        order: the order in which the vertices appear in time (to fit the second condition)
        adj_matrix: returns an adjacency matrix if True, a Graph if False
    """
    M = nx.to_scipy_sparse_matrix(G, nodelist=order, weight=None, format='lil')

    # Close the graph
    nodes = sp.sparse.find(M)
    for i, j in sorted(zip(nodes[0], nodes[1]), key=lambda a: a[0])[::-1]:
        M[i] = np.logical_or(M[i].toarray(), M[j].toarray())

    # Return in the proper format
    if adj_matrix is True:
        return M.tocsr()
    else:
        GT = G.copy()
        edges = sp.sparse.find(M)
        for k in range(len(edges[0])):
            i, j = edges[0][k], edges[1][k]
            GT.add_edge(order[i], order[j])
        return GT


def transitive_reduction_np(G, order=None, adj_matrix=False):
    """
    Return the transitive reduction of a given directed graph using NumPy's arrays.

    Based on Aho, A. V., Garey, M. R., Ullman, J. D. The Transitive
    Reduction of a Directed Graph. SIAM Journal on Computing 1, 131–137
    (1972).

    Arguments:
        order: the order in which the vertices appear in time (to fit the second condition). If None, uses the same order as in the graph
        adj_matrix: returns an adjacency matrix if True, a Graph if False
    """
    # Transitive closure
    MT = transitive_closure_np(G, order, True)

    # Reorganize the adjacency matrix
    M = nx.to_numpy_matrix(G, nodelist=order, weight=None, dtype=np.int8)

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


def transitive_closure_np(G, order=None, adj_matrix=False):
    """
    Return the transivite closure of a graph using NumPy's Arrays.

    This method assumes that your graph is:
        1. Directed acyclic
        2. Organized so that a node can only interact with a node
        positioned after it, in the adjacency matrix
    Arguments:
        order: the order in which the vertices appear in time (to fit the second condition)
        adj_matrix: returns an adjacency matrix if True, a Graph if False
    """
    M = nx.to_numpy_matrix(G, nodelist=order, weight=None, dtype=np.int8)

    # Close the graph
    nodes = np.where(M == 1)
    for i, j in sorted(zip(nodes[0], nodes[1]), key=lambda a: a[0])[::-1]:
        M[i] = np.logical_or(M[i], M[j])

    # Return in the proper format
    if adj_matrix is True:
        return M.astype(bool)
    else:
        GT = G.copy()
        for i, j in np.array(np.where(M == 1)).T:
            GT.add_edge(order[i], order[j])
        return GT
