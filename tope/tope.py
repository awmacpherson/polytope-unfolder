import polytope, numpy as np
from pypoman import compute_polytope_halfspaces

#import networkx as nx
#from networkx import minimum_spanning_arborescence, from_edgelist

from loguru import logger
import itertools, functools, collections
from dataclasses import dataclass
from typing import Any

from .orth import *

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
    metadata: list[list[dict[str, Any]]] = None

    @property
    def dim(self):
        return self.vertices.shape[1]

    @property
    def vindex(self):
        return np.arange(self.vertices.shape[0])

    @classmethod
    def from_vertices(cls, vertices):
        vertices = np.array(vertices)
        logger.debug(f"Computing from set of {vertices.shape[0]} vertices.")

        _A, _b = compute_polytope_halfspaces(vertices)
        assert _A.ndim == 2
        assert _A.shape[1] == vertices.shape[1]
        assert _b.ndim == 1
        assert _b.shape[0] == _A.shape[0]
        logger.debug(f"Found {len(_b)} facets.")

        dim = vertices.shape[1]
        nverts = vertices.shape[0]
        faces: list[list[set]] = [[] for _ in range(dim)]

        # vertices we already know
        faces [0]            = [{n} for n in range(nverts)]

        # facets we compute directly from supporting hyperplanes
        faces [-1] = [
            intersect_set_with_affine_subspace(vertices, A, b) 
            for A, b in zip(_A, _b)
        ]

        eliminate_repetitions(faces [-1])

        # now do codimension 2 faces down to edges
        for k in range(dim-2, 0, -1):
            #logger.debug(f"Extracting {k}-diml faces...")
            for facet_i in faces[-1]:
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

        # add metadata lists
        metadata = []
        for l in faces:
            metadata.append([{}] * len(l))
        
        return cls(vertices, faces, metadata)

# DEPRECATED 
        newtope = cls(vertices, faces)
        newtope.metadata = metadata
        return newtope
# /DEPRECATED

    def verify_face(self, i, k=-1):
        k = self.dim + k if k < 0 else k
        main:   set[int]    = self.faces[k][i]
        main_l: list[int]   = list(main)
        vertices:   np.ndarray = self.vertices[main_l]
        
        logger.info(f"{k}-face {i} has affine dimension {affine_span_dim(vertices)}.") 
        #assert affine_span_codim(vertices) == self.dim - k
    
    def get_face(self, i, k=-1):
        main:   set[int]    = self.faces[k][i]
        main_l: list[int]   = sorted(main) # guarantee ordering of vertices
        vertices:   np.ndarray = self.vertices[main_l]

        faces:  list[list[int]] = [[] for _ in self.faces[:k]]
        labels: list[list[int]] = [[] for _ in self.faces[:k]]
        for j, self_j_faces in enumerate(self.faces[:k]):
            for n, face in enumerate(self_j_faces):
                if face.issubset(main):
                    faces[j].append(set([main_l.index(v) for v in face]))
                    labels[j].append(n)

        P = Tope(vertices, faces)

        # let's label the faces too!
        P.labels = labels
        return P

    def get_facet(self, i):
        inward_normal = self.vertices.mean(axis=0) - \
                self.vertices[list(self.faces[-1][i])].mean(axis=0)
        return self.get_face(i, -1).in_own_span(orientation = inward_normal)
    
    def in_own_span(self, orientation=None):
        v, _ = in_own_span(self.vertices - self.vertices.mean(axis=0), orientation)
        P = Tope(v, self.faces)
        if hasattr(self, "labels"):
            P.labels = self.labels
        return P


    def interface(self, i, j):
        """Return the intersection of two facets if codimension two or None."""
        s = set.intersection(self.faces[-1][i], self.faces[-1][j])
        #logger.debug(f"Found intersection {s}.")
        return s if s in self.faces[-2] else None


