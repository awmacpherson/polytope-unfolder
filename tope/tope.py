import polytope, numpy as np
from pypoman import compute_polytope_halfspaces

#import networkx as nx
#from networkx import minimum_spanning_arborescence, from_edgelist

from loguru import logger
import itertools, functools, collections
from dataclasses import dataclass
from typing import Any, Iterable
from copy import deepcopy

from .orth import *
from .graph import Graph

def eliminate_repetitions(l: list[set]):
    popidx = []
    for n, s in enumerate(l):
        for j in range(len(l)-1, n, -1):
            # pop from the end to avoid fucking things up
            if s == l[j]:
                l.pop(j)

@dataclass
class Tope:
    vertices: np.ndarray
    faces: list[list[set[int]]]
    meta: list[list[dict[str, Any]]] = None

    def __post_init__(self):
        if self.faces is None:
            self.faces = [[] for _ in range(self.dim+1)]
        if self.meta is None:
            self.meta  = [[] for _ in range(self.dim+1)]

    @property
    def dim(self):
        return self.vertices.shape[1]

    @property
    def vindex(self):
        return np.arange(self.vertices.shape[0])


    ### ITERATORS ###

    def iter_faces(self, dim=None, yield_as="vertices") -> Iterable[set[int]]:
        """
        Return iterable over faces as sets of integer indices.
        If dim is None, iterate over faces of all dimensions.
        If dim is negative, interpret as codimension.
        """
        return self.iter_all_faces() if dim is None else \
                self.faces[dim if dim >= 0 else self.dim-dim]

    def iter_all_faces(self) -> Iterable[set[int]]:
        return (face for n_faces in self.faces for face in n_faces)

    def iter_faces_as_topes(self, dim=None):
        raise NotImplementedError

    def iter_faces_as_vertices(self, dim=None) -> Iterable[np.ndarray]: 
        # used to get edges for final plot
        return map(lambda l: self.vertices[l], map(sorted, self.iter_faces(dim=dim)))

    def iter_meta(self, dim=None, key=None): # used in apply_to_meta()
        if dim is None: return self.iter_all_meta(key)# iterate over faces of all dims
        if key is None: return self.meta[dim if dim >= 0 else self.dim-dim]
        return (meta[key] for meta in self.meta[dim if dim >= 0 else self.dim-dim])

    def iter_meta_key(self, key, dim=None):
        return self.iter_all_meta_key(key) if dim is None else \
                (meta[key] for meta in self.meta[dim])

    def iter_all_meta(self, key=None):
        return (meta for n_meta in self.meta for meta in n_meta)\
                if key is None else self.iter_all_meta_key(key)

    def iter_all_meta_key(self, key):
        return (meta[key] for n_meta in self.meta for meta in n_meta)

    def enumerate_all_faces(self):
        return ((n, i, face) for n, n_faces in enumerate(self.faces) \
                for i, face in enumerate(n_faces))

    def enumerate_all_faces_meta(self): # used in get_face
        return ((n, i, face, self.meta[n][i]) \
                for n, i, face in self.enumerate_all_faces())

    @classmethod
    def from_vertices(cls, vertices):
        vertices = np.array(vertices)
        logger.debug(f"Computing from set of {vertices.shape[0]} vertices.")

        
        _A, _b = compute_polytope_halfspaces(vertices)
        # The following all succeed:
        # assert _A.ndim == 2
        # assert _A.shape[1] == vertices.shape[1]
        # assert _b.ndim == 1
        # assert _b.shape[0] == _A.shape[0]
        logger.debug(f"Found {len(_b)} facets.")

        dim = vertices.shape[1]
        nverts = vertices.shape[0]
        faces: list[list[set]] = [[] for _ in range(dim+1)]

        # vertices we already know
        faces [0]   = [{n} for n in range(nverts)]
        # top cell we already know
        faces [dim] = [set(range(nverts))]

        # facets we compute directly from supporting hyperplanes
        faces [dim-1] = [
            intersect_set_with_affine_subspace(vertices, A, b) 
            for A, b in zip(_A, _b)
        ]

        eliminate_repetitions(faces [dim-1])

        # now do codimension 2 faces down to edges
        for k in range(dim-2, 0, -1):
            #logger.debug(f"Extracting {k}-diml faces...")
            for facet_i in faces[dim-1]:
                for face_j in faces[k+1]:
                    if not face_j.issubset(facet_i):
                        #logger.debug(f"Found {face_j.intersection(facet_i)}.")
                        faces[k].append(face_j.intersection(facet_i))

            removals = set()
            for i, face_i in enumerate(faces[k]):
                for j, face_j in enumerate(faces[k][i+1:]):
                    if face_i.issubset(face_j):
                        #logger.debug(f"Removing {face_i}.")
                        removals.add(i)
                        break
                    if face_j.issubset(face_i):
                        #logger.debug(f"Removing {face_j}.")
                        removals.add(j+i+1)

            #logger.debug(f"Processing removals {removals}.")
            for i in sorted(removals, reverse=True):
                faces[k].pop(i)

        logger.debug(f"Finished processing faces:\n {faces}")

        # add meta lists
        meta = []
        for l in faces:
            meta.append([{} for _ in l])
        
        return cls(vertices, faces, meta)

    def get_face(self, i, k=-1):
        """
        Returns a Tope object consisting of all subfaces of a given face.
        Metadata is preserved. Makes no guarantees about orientation. 
        """
        k = self.dim + k if k < 0 else k
        target_face:        set[int]    = self.faces[k][i]
        target_face_idx:    list[int]   = sorted(target_face)

        vertices:   np.ndarray = self.vertices[target_face_idx] # vertices of face

        Q = Tope(vertices, None, None)

        for j, i, face, meta in self.enumerate_all_faces_meta():
            if face.issubset(target_face):
                Q.faces[j].append( { target_face_idx.index(v) for v in face } )
                Q.meta[j].append( deepcopy(meta) )

        return Q


# DEPRECATED

    def get_face_with_remap(self, i, k=-1):
        """
        Returns a Tope object consisting of all subfaces of a given face.
        Metadata is preserved. Makes no guarantees about orientation. 
        """
        k = self.dim + k if k < 0 else k
        target_face:        set[int]    = self.faces[k][i]
        target_face_idx:    list[int]   = sorted(target_face)

        vertices:   np.ndarray = self.vertices[target_face_idx] # vertices of face

        Q = Tope(vertices, None, None)

        for j, i, face, meta in self.enumerate_all_faces_meta():
            if face.issubset(target_face):
                Q.faces[j].append( { target_face_idx.index(v) for v in face } )
                remap = index_like(Q)

        return Q

# /DEPRECATED

    def __eq__(self, other):
        return (self.vertices == other.vertices).all() and self.faces == other.faces\
                and self.meta == other.meta

    def get_facet(self, i, meta_keys = []):
        inward_normal = self.vertices.mean(axis=0) - \
                self.vertices[list(self.faces[self.dim-1][i])].mean(axis=0)
        return self.get_face(i, -1).in_own_span(orientation = inward_normal,
                meta_keys = meta_keys)
    
    def in_own_span(self, orientation=None, meta_keys = []):
        """
        Return self with vertices expressed in a basis for its own span.
        Apply same change of basis to chosen meta keys.
        """
        origin = self.vertices.mean(axis=0)
        v, basis = in_own_span(self.vertices - origin, orientation)

        P = Tope(v, self.faces, self.meta)

        for key in meta_keys:
            P.apply_to(lambda x: (x-origin) @ basis.T, key)
        
        return P

    def interface(self, i, j, return_as="set") -> set:
        """
        Return the intersection of two facets if codimension two or None.
        Used in get_facet_graph.
        """
        s = set.intersection(self.faces[self.dim-1][i], self.faces[self.dim-1][j])
        #logger.debug(f"Found intersection {s}.")
        return s if s in self.faces[self.dim-2] else None

    def facet_graph(self) -> Graph:
        """
        Return an intersection graph of facets with nodes labelled by the
        corresponding Tope objects and edges by sets of indices into 
        self.vertices.
        """
        nodes = range(len(self.faces[self.dim - 1]))
        node_labels = {i: self.get_face(i, k=self.dim-1) for i in nodes}
        return Graph.from_pairing(nodes, self.interface, node_labels=node_labels)

    def save_index(self, key = "index"):
        """
        Save indices of faces into meta under <key> so that they are preserved on 
        passing to sub-Topes. Overwrites existing meta for <key>.
        """
        for k, i, _, meta in self.enumerate_all_faces_meta():
            meta[key] = i


    def apply_to(self, transform, key):
        """
        Apply transform to all meta entries under key.
        """
        for meta in self.iter_meta():
            if key in meta:
                meta[key] = transform(meta[key])

# move outside class

    def cut_2faces_with_hyperplanes(self, hyperplanes) -> list[np.ndarray]: # [2][2]float
        """
        Goes through a list of hyperplanes, recording intersections with 2-faces.
        Attaches it to those 2-faces as meta under "cuts."
        """
        # this operation can't easily be stacked and carried out in numpy
        # because of branching in intersect_*_with_hyperplane methods.
        # or maybe it could if I did it a cleverer way.

        tmp = []
        for i, face in enumerate(self.faces[2]):
            X = self.get_face(i,2)
            for H in hyperplanes: # [self.dim], [self.dim]
                logger.debug(f"Intersecting with hyperplane {H}.")
                q = intersect_polygon_with_hyperplane(X, H) # returns list of [2][dim]float
                if q is not None:
                    tmp.append(q)
            self.meta[2][i]["cuts"] = np.stack(tmp) if tmp else np.zeros((0,2,self.dim))
            tmp.clear()

