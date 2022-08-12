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
            metadata.append([{} for _ in l])
        
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

    def save_index(self, key = "index"):
        """
        Save indices of faces into metadata under <key> so that they are preserved on 
        passing to sub-Topes. Overwrites existing metadata for <key>.
        """
        for k, faces_list in enumerate(self.faces):
            for i, face in enumerate(faces_list):
                self.metadata[k][i][key] = i

    def get_face(self, i, k=-1):
        """
        Returns a Tope object consisting of all subfaces of a given face.
        Metadata is preserved. Makes no guarantees about orientation. 
        """
        target_face:   set[int]    = self.faces[k][i]
        main_l: list[int]   = sorted(target_face) # sorted list of indices into self.vertices
        # main_l: range(len(target_face)) -> range(len(tope))

        vertices:   np.ndarray = self.vertices[main_l] # vertices of face

        # note: [[]] * N creates a list of N references to the same underlying list!
        # So we have to use this list comprehension.
        faces:  list[list[int]]             = [[] for _ in self.faces[:k]]
        meta:   list[list[dict[str, Any]]]  = [[] for _ in self.faces[:k]]

        for j, all_j_faces in enumerate(self.faces[:k]):
            for n, face in enumerate(all_j_faces):
                if face.issubset(target_face):
                    # convert indices into range(len(tope)) 
                    # to indices into range(len(target_face))
                    faces[j].append(set([main_l.index(v) for v in face]))

                    # Copy metadata
                    meta[j] .append(self.metadata[j][n])

        return Tope(vertices, faces, meta)

    def __eq__(self, other):
        return (self.vertices == other.vertices).all() and self.faces == other.faces\
                and self.metadata == other.metadata

    def _get_face(self, i, k=-1):
        """
        Returns a Tope object consisting of all subfaces of a given face.
        Metadata is preserved. If there is no "label" metadata, add it with
        an index into 
        """
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

        P.labels = labels # DEPRECATE --- to be removed
        return P

    def get_facet(self, i, metadata_keys = []):
        inward_normal = self.vertices.mean(axis=0) - \
                self.vertices[list(self.faces[-1][i])].mean(axis=0)
        return self.get_face(i, -1).in_own_span(orientation = inward_normal,
                metadata_keys = metadata_keys)
    
    def in_own_span(self, orientation=None, metadata_keys = []):
        """
        Return self with vertices expressed in a basis for its own span.
        Apply same change of basis to chosen metadata keys.
        """
        v, basis = in_own_span(self.vertices - self.vertices.mean(axis=0), orientation)

        P = Tope(v, self.faces, self.metadata)

        for key in metadata_keys:
            P.apply_to(lambda x: x @ basis.T, key)
        
        return P

    def interface(self, i, j):
        """
        Return the intersection of two facets if codimension two or None.
        Used in get_facet_graph.
        """
        s = set.intersection(self.faces[-1][i], self.faces[-1][j])
        #logger.debug(f"Found intersection {s}.")
        return s if s in self.faces[-2] else None

    def apply_to(self, transform, key):
        """
        Apply transform to all metadata entries under key.
        """
        for k in range(self.dim):
            for i in range(len(self.faces[k])):
                if key in self.metadata[k][i]:
                    self.metadata[k][i][key] = transform(self.metadata[k][i][key])

    def cut_2faces_with_hyperplanes(self, hyperplanes) -> list[np.ndarray]: # [2][2]float
        """
        Goes through a list of hyperplanes, recording intersections with 2-faces.
        Attaches it to those 2-faces as metadata under "cuts."
        """
        # this operation can't easily be stacked and carried out in numpy
        # because of branching in intersect_*_with_hyperplane methods.
        # or maybe it could if I did it a cleverer way.

        tmp = []
        for i, face in enumerate(self.faces[2]):
            X = self.get_face(i,2)
            for H in hyperplanes: # [self.dim], [self.dim]
                q = intersect_polygon_with_hyperplane(X, H)
                if q is not None:
                    tmp.append(q)
            self.metadata[2][i]["cuts"] = np.stack(tmp)
            tmp.clear()

