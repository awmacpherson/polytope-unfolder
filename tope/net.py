from dataclasses import dataclass
import numpy as np
from loguru import logger
from .graph import Graph
from .orth import rotate_into_hyperplane, in_own_span, affine_span_dim, fold_into_hyperplane

FLOAT_ERR = 0.000001

# Move inside Tope class
def get_facet_graph(P) -> Graph:
    """
    Create Graph object with nodes 0, ..., #{facets of P} and edges labelled
    by the intersection of two facets, considered as a set of indices into
    P.vertices.
    """
    node_labels = dict(enumerate(P.faces[P.dim-1]))
    return Graph.from_pairing(node_labels, P.interface, node_labels=node_labels)

import functools
from typing import Callable

@dataclass
class Net2:
    """
    Wrapper class for a Graph with nodes labelled by Topes and edges labelled
    by the intersections thereof.
    """
    tree:   Graph

    @property
    def cells(self):
        return self.tree.node_labels

    @classmethod
    def from_tope(cls, P):
        tree = P.facet_graph().get_spanning_tree()
        return cls(P, tree)

    @property
    def cell_dim(self):
        """
        Return dimension of an arbitrarily chosen cell.
        Raise StopIterationError if there are no cells.
        """
        return self.cells.values().iter().next().dim

    def unfold(self, start = None): # modify facets dict in place
        start = self.tree.root if start is None else start

        for node in self.tree.children[start]:
            self.unfold(start=node)

            F0 = self.cells[start]
            F1 = self.cells[node]
            vertices = np.concatenate([F0.vertices, F1.vertices])

            #rotation, offset = rotate_into_hyperplane(self.tope.vertices, F0, F1)
            rotation, offset = fold_into_hyperplane(self.tope.vertices, F0, F1)
            for i in self.tree.iter_from(node):
                self.facets[i] = ((self.facets[i] - offset) @ rotation) + offset

        return self

    ### Iterators ###
    
    def iter_cells(self): # needed?
        return self.cells.values()

    # These just concatenate iterators from each cell #

    def iter_faces(self, dim=None, yield_as="vertices"): # may have repetitions
        return (cell.vertices[idx] \
                for cell in self.cells.values() \
                for idx in cell.iter_faces(dim=dim))

    def iter_meta(self, dim=None, key=None):
        return (v for cell in self.cells.values() for v in cell.iter_meta(dim=dim, key=key))

class Net:
    # TODO Replace tope in initializer with cells: list[Tope]
    # and add classmethod from_tope(Tope, Graph)
    def __init__(self, P, T: Graph):
        self.tope = P
        self.tree: Graph = T # facet tree labelled by Pow(num_vertices)

        # mutable mapping of Topes
        self.facets = T.node_labels

    def iter_faces_as_arrays(self, dim):
        return (face for facet in self.facets.values() for face in facet.iter_faces_as_arrays(dim))

    def iter_edges_as_arrays(self):
        """
        Iterate over all edges of all cells in a form that can be consumed by
        mpl.collections.LineCollection.
        """
        return self.iter_faces_as_arrays(1)

    iter_edges = iter_edges_as_arrays

    @classmethod
    def from_tope(cls, P):
        T = P.facet_graph().get_spanning_tree()
        return cls(P, T)

    def unfold(self, start = None): # modify facets dict in place
        start = self.tree.root if start is None else start

        for node in self.tree.children[start]:
            self.unfold(start=node)

            F0 = self.tope.faces[self.tope.dim-1][start]
            F1 = self.tope.faces[self.tope.dim-1][node]

            rotation, offset = fold_into_hyperplane(self.tope.vertices, F0, F1)
            #rotation, offset = rotate_into_hyperplane(self.tope.vertices, F0, F1)
            for i in self.tree.iter_from(node):
                self.facets[i].vertices = ((self.facets[i].vertices - offset) @ rotation) + offset

        return self

    def unfold_with_meta(self, start = None, meta_keys = []): 
        # as unfold(), but also modify in-place meta with keys listed in decoration
        start = self.tree.root if start is None else start

        for node in self.tree.children[start]:
            self.unfold_with_meta(start=node, meta_keys = meta_keys)

            F0 = self.tope.faces[self.tope.dim-1][start]
            F1 = self.tope.faces[self.tope.dim-1][node]

            rotation, offset = rotate_into_hyperplane(self.tope.vertices, F0, F1)
            rotate = lambda x : ((x - offset) @ rotation) + offset
            
            for i in self.tree.iter_from(node):
                self.facets[i].vertices = rotate(self.facets[i].vertices)
                for k in meta_keys:
                    self.tope.meta[self.tope.dim-1][i][k] = \
                        rotate(self.tope.meta[self.tope.dim-1][i][k])

        return self

    def in_own_span(N, meta_keys = []):
        """
        Reencode vertices in basis for their own affine span. Apply to unfolded net. 
        Orientation is normalised so that taking the inward-pointing normal as the 
        last basis vector for the ambient space is oriented w.r.t. the standard basis.
        """
        root_facet = list(N.tope.faces[N.tope.dim-1][N.tree.root])
        ref_pt = N.tope.vertices[root_facet].mean(axis=0)
        inward_normal = N.tope.vertices.mean(axis=0) - ref_pt

        facet_index = sorted(N.facets.keys())
        
        offsets = [0] + [len(N.facets[i].vertices) for i in facet_index]
        offsets = np.cumsum(offsets)
        all_vertices = np.concatenate([N.facets[i].vertices for i in facet_index])
        logger.debug(f"vertices shape: {all_vertices.shape}")
        all_vertices, basis = in_own_span(
                all_vertices - ref_pt, 
                orientation=inward_normal
                )

        for key in meta_keys:
            N.tope.apply_to(lambda x : (x - ref_pt) @ basis.T, key)

        # Need to reflect in one axis if orientation of root face is wrong.
#        if np.linalg.det(np.c_[basis.T, inward_normal]) < 1:
#            all_vertices[:,0] = -all_vertices[:,0]
        
        for i in facet_index:
            N.facets[i].vertices = all_vertices[offsets[i]:offsets[i+1]]

        return N

