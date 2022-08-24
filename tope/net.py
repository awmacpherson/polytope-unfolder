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
    # TODO Replace tope in initializer with cells: list[Tope]
    # and add classmethod from_tope(Tope, Graph)
    def __init__(self, P: Tope, T: Graph):
        self.tope: Tope = P
        self.tree: Graph = T # facet tree labelled by Pow(num_vertices)

        # mutable
        self.facets = {i: self.tope.vertices[sorted(T.node_labels[i])] for i in T.nodes}

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

    def unfold_with_metadata(self, start = None, meta_keys = []): 
        # as unfold(), but also modify in-place metadata with keys listed in decoration
        start = self.tree.root if start is None else start

        for node in self.tree.children[start]:
            self.unfold_with_metadata(start=node, meta_keys = meta_keys)

            F0 = self.tope.faces[-1][start]
            F1 = self.tope.faces[-1][node]

            rotation, offset = rotate_into_hyperplane(self.tope.vertices, F0, F1)
            rotate = lambda x : ((x - offset) @ rotation) + offset
            
            for i in self.tree.iter_from(node):
                self.facets[i] = rotate(self.facets[i])
                for k in meta_keys:
                    self.tope.metadata[-1][i][k] = rotate(self.tope.metadata[-1][i][k])

        return self

    def in_own_span(N, meta_keys = []):
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
        all_vertices, basis = in_own_span(
                all_vertices - ref_pt, 
                orientation=inward_normal
                )

        for key in meta_keys:
            N.tope.apply_to(lambda x : (x - ref_pt) @ basis.T, key)

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
    
        
