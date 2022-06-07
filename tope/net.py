from dataclasses import dataclass
import numpy as np
from loguru import logger
from .graph import Graph
from .tope import Tope
from .orth import rotate_into_hyperplane

FLOAT_ERR = 0.000001

def get_facet_graph(P: Tope) -> Graph:
    node_labels = dict(enumerate(P.faces[-1]))
    return Graph.from_pairing(
        node_labels, 
        P.interface, 
        node_labels=node_labels 
    )

def affine_span(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Assume that A spans a codimension one affine subspace and 
    that the ambient space has dimension at most 3.
    This second assumption is necessary to conclude that *any* n vectors
    span an (n-1)d affine hyperplane.
    """
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


import functools
from typing import Callable

class Net:
    def __init__(self, P: Tope, T: Graph):
        self.tope: Tope = P
        self.tree: Graph = T # facet tree labelled by Pow(N)
        #self.vertices: np.ndarray = np.array(P.vertices)
        #self.embedded_vertices: np.ndarray = self.vertices @ emb

        # mutable
        self.facets = {i: self.tope.vertices[list(T.node_labels[i])] for i in T.nodes}

    def unfold(self, start = None): # modify facets dict in place
        start = self.tree.root if start is None else start

        for node in self.tree.children[start]:
            self.unfold(start=node)

            F0 = self.tope.faces[-1][start]
            F1 = self.tope.faces[-1][node]
            I = set.intersection(F0,F1)

            rotation, offset = rotate_into_hyperplane(
                self.tope.vertices, F0, F1, I
            )
            self.apply_recurse(lambda X : ((X-offset)@rotation.T)+offset, start=node)

    def apply_recurse(self, func: Callable, start=None):
        start = self.tree.root if start is None else start
        self.facets[start] = func(self.facets[start])
        for node in self.tree.children[start]:
            self.apply_recurse(func, start=node)

        

# DEPRECATED

    def interface(self, i, j): # doesn't update during unfolding
        return self.embedded_vertices[list(self.tree.edge_labels[(i,j)])]

    def interface_h(self, i, j):
        return affine_span(self.interface(i,j))

    def folded(self, i, j):
        "Check if the two cells associated to an edge are folded."
        eq, offset = affine_span(self.interface(i,j))
        return sign(self.facets[i], eq, offset) == sign(self.facets[j], eq, offset)


    def _unfold(self, *pretransform, src = None): # modify facets dict in place
        src = self.tree.root if src is None else src
        for inc in self.children(src):

            transform = pretransform + [get_refl(inc.interstice)] \
                    if inc.is_folded else pretransform
            self.unfold(*transform, src=inc.facet2)

