from dataclasses import dataclass
import numpy as np
from loguru import logger
from .graph import Graph
from .tope import Tope
from .orth import rotate_into_hyperplane, in_own_span, affine_span_dim

FLOAT_ERR = 0.000001

def get_facet_graph(P: Tope) -> Graph:
    node_labels = dict(enumerate(P.faces[-1]))
    return Graph.from_pairing(node_labels, P.interface, node_labels=node_labels)

def put_in_own_span(N):
    """
    Reencode vertices in basis for their own affine span. Apply to unfolded net. 
    Orientation is normalised so that taking the inward-pointing normal as the 
    last basis vector for the ambient space is oriented w.r.t. the standard basis.
    """
    root_facet = list(N.tope.faces[-1][N.tree.root])
    ref_pt = N.tope.vertices[root_facet].mean(axis=0)
    
    offsets = [0] + [len(vertices) for vertices in N.facets.values()]
    offsets = np.cumsum(offsets)
    all_vertices = np.concatenate(list(N.facets.values()))
    all_vertices, basis = in_own_span(all_vertices - ref_pt)
                       
    # Need to reflect in one axis if orientation of root face is wrong.
    inward_normal = N.tope.vertices.mean(axis=0) - ref_pt
    if np.linalg.det(np.c_[basis.T, inward_normal]) < 1:
        all_vertices[:,0] = -all_vertices[:,0]
    
    for i in N.facets:
        N.facets[i] = all_vertices[offsets[i]:offsets[i+1]]

import functools
from typing import Callable

class Net:
    def __init__(self, P: Tope, T: Graph):
        self.tope: Tope = P
        self.tree: Graph = T # facet tree labelled by Pow(num_vertices)

        # mutable
        self.facets = {i: self.tope.vertices[list(T.node_labels[i])] for i in T.nodes}

    def unfold(self, start = None): # modify facets dict in place
        start = self.tree.root if start is None else start

        for node in self.tree.children[start]:
            self.unfold(start=node)

            F0 = self.tope.faces[-1][start]
            F1 = self.tope.faces[-1][node]

            rotation, offset = rotate_into_hyperplane(self.tope.vertices, F0, F1)
            for i in self.tree.iter_from(node):
                self.facets[i] = ((self.facets[i] - offset) @ rotation) + offset

        return self

    def in_own_span(N):
        """
        Reencode vertices in basis for their own affine span. Apply to unfolded net. 
        Orientation is normalised so that taking the inward-pointing normal as the 
        last basis vector for the ambient space is oriented w.r.t. the standard basis.
        """
        root_facet = list(N.tope.faces[-1][N.tree.root])
        ref_pt = N.tope.vertices[root_facet].mean(axis=0)
        inward_normal = N.tope.vertices.mean(axis=0) - ref_pt
        
        offsets = [0] + [len(vertices) for vertices in N.facets.values()]
        offsets = np.cumsum(offsets)
        all_vertices = np.concatenate(list(N.facets.values()))
        all_vertices, basis = in_own_span(all_vertices - ref_pt, orientation=inward_normal)
                           
        # Need to reflect in one axis if orientation of root face is wrong.
#        if np.linalg.det(np.c_[basis.T, inward_normal]) < 1:
#            all_vertices[:,0] = -all_vertices[:,0]
        
        for i in N.facets:
            N.facets[i] = all_vertices[offsets[i]:offsets[i+1]]

        return N

# DEPRECATED

            #self.apply_recurse(lambda X : ((X-offset)@rotation.T)+offset, start=node)

    def apply_recurse(self, func: Callable, start=None):
        start = self.tree.root if start is None else start
        self.facets[start] = func(self.facets[start])
        for node in self.tree.children[start]:
            self.apply_recurse(func, start=node)
    
        

"""
def affine_span(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ""
    Assume that A spans a codimension one affine subspace and 
    that the ambient space has dimension at most 3.
    This second assumption is necessary to conclude that *any* n vectors
    span an (n-1)d affine hyperplane.
    ""
    N = A.shape[1]
    
    offset = np.sum(A[:N], axis=0) / N # do this to avoid divide by zero shenanigans
    
    A_lin = A[:N] - offset # linearise
    eigval, eigvec = np.linalg.eig(A_lin) # complex dtype
    logger.debug(f"Found eigenvalues {eigval} and eigenvectors {eigvec}")
    i = np.abs(eigval).argmin()
    if eigval[i] > FLOAT_ERR: # should get zero eigenvalue (+ floating point error)
        raise Exception("Oh no! Input wasn't contained in a hyperplane!")
    return eigvec[:,i].real, offset

def sign(X, eq, offset) -> bool:
    sgn = (X-offset) @ eq
    if not ((sgn>-FLOAT_ERR).all() + (sgn<FLOAT_ERR).all()) % 2: raise
    return (sgn>-FLOAT_ERR).all()

def apply_refl(X: np.ndarray, eq, offset):
    # X.shape = (N, 3)
    # eq.shape = offset.shape = (3,)
    # complains if X is rank one
    # apparently has really bad error (3rd s.f.)
    return X - np.tensordot(
        2 * np.dot(X-offset, eq) / np.dot(eq, eq),
        eq,
        axes=0
    )
"""
