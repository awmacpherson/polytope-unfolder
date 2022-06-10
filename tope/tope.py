import polytope, numpy as np
from pypoman import compute_polytope_halfspaces

#import networkx as nx
#from networkx import minimum_spanning_arborescence, from_edgelist

from loguru import logger
import itertools, functools, collections
from dataclasses import dataclass

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

#        _tope = polytope.qhull(np.array(vertices)) # converts to float
#        logger.debug(f"Convex hull has {_tope.vertices.shape[0]} vertices.")

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

#        for i, face in enumerate(faces[-1]):
#            logger.info(f"{dim-1}-face {i} has affine codimension {affine_span_codim(vertices[list(face)])}.") 

        eliminate_repetitions(faces [-1])

        # ALTERNATIVE ALGORITHM:
        # compute facets directly as maximal extremal sets of vertices
        
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
        
        return cls(vertices, faces)

    def verify_face(self, i, k=-1):
        k = self.dim + k if k < 0 else k
        main:   set[int]    = self.faces[k][i]
        main_l: list[int]   = list(main)
        vertices:   np.ndarray = self.vertices[main_l]
        
        logger.info(f"{k}-face {i} has affine dimension {affine_span_dim(vertices)}.") 
        #assert affine_span_codim(vertices) == self.dim - k
        

    def get_face(self, i, k=-1):
        main:   set[int]    = self.faces[k][i]
        main_l: list[int]   = list(main)
        vertices:   np.ndarray = self.vertices[main_l]

#        faces_labels = [
#            [
#                [set([main_l.index(v) for v in face]), n]
#                for n, face in enumerate(j_faces) if face.issubset(main)
#            ]
#            for j_faces in self.faces[:k] 
#        ]
#        faces, labels = zip((zip(j_faces_labels) for j_faces_labels in faces_labels))

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
    
    @property
    def in_own_span(self):
        return Tope(in_own_span(self.vertices - self.vertices.mean(axis=0)), self.faces)

    def ___init__(self, vertices):
        vertices = np.array(vertices)
        logger.debug(f"Computing from set of {vertices.shape[0]} vertices.")
        self.vertices = vertices
        self.polytope = polytope.qhull(np.array(vertices)) # converts to float
        logger.debug(f"Constructed polytope supports.")
        #logger.debug(f"A: \n{self.polytope.A}")
        #logger.debug(f"b: \n{self.polytope.b}")
        logger.debug(f"Convex hull has {self.polytope.vertices.shape[0]} vertices.")

        self.faces: list[list[set]] = [[] for _ in range(self.dim)]
        # kth entry is list of k-dimensional faces as sets of vertex indices

        # vertices we already know
        self.faces [0]            = [{n} for n in self.vindex]

        # facets we compute directly from supporting hyperplanes
        self.faces [self.dim-1] = [
            set(
                self.vindex [np.abs(h @ self.polytope.vertices.T - c) < 0.000001] 
                # SELECTION MAY RESULT IN ROUNDING ERRORS
            )
            for h, c in zip(self.polytope.A, self.polytope.b)
        ]

        eliminate_repetitions(self.faces [self.dim-1])

        # ALTERNATIVE ALGORITHM:
        # compute facets directly as maximal extremal sets of vertices
        
        # now do codimension 2 faces down to edges
        for k in range(self.dim-2, 0, -1):
            logger.debug(f"Extracting {k}-diml faces...")
            for facet_i in self.faces[-1]:
                for face_j in self.faces[k+1]:
                    if not face_j.issubset(facet_i):
                        #logger.debug(f"Found {face_j.intersection(facet_i)}.")
                        self.faces[k].append(face_j.intersection(facet_i))

            removals = set()
            for i, face_i in enumerate(self.faces[k]):
                for j, face_j in enumerate(self.faces[k][i+1:]):
                    if face_i.issubset(face_j):
                        #logger.debug(f"Removing {face_i}.")
                        removals.add(i)
                        break
                    if face_j.issubset(face_i):
                        #logger.debug(f"Removing {face_j}.")
                        removals.add(j+i+1)

            logger.debug(f"Processing removals {removals}.")
            for i in sorted(removals, reverse=True):
                self.faces[k].pop(i)

        logger.debug(f"Finished processing faces:\n {self.faces}")

    def interface(self, i, j):
        """Return the intersection of two facets if codimension two or None."""
        s = set.intersection(self.faces[-1][i], self.faces[-1][j])
        #logger.debug(f"Found intersection {s}.")
        return s if s in self.faces[-2] else None


# --- DEPRECATED -------------------------------------------

    def __bad_init__(self, vertices):
        self.vertices = vertices
        self.polytope = polytope.qhull(np.array(vertices)) # converts to float
        logger.debug(f"Constructed polytope supports.")
        logger.debug(f"A: \n{self.polytope.A}")
        logger.debug(f"b: \n{self.polytope.b}")

        self.ndims = self.polytope.A.shape[1]
        self.vindex = np.arange(len(vertices))

        self.faces: list[list[set]] = [[] for _ in range(self.ndims)]
        # kth entry is list of k-dimensional faces as sets of vertex indices

        # vertices we already know
        self.faces [0]            = [{n} for n in self.vindex]

        # facets we compute directly from supporting hyperplanes
        self.faces [self.ndims-1] = [
            set(
                self.vindex [np.abs(h @ self.polytope.vertices.T - c) < 0.000001] 
                # SELECTION MAY RESULT IN ROUNDING ERRORS
            )
            for h, c in zip(self.polytope.A, self.polytope.b)
        ]

        eliminate_repetitions(self.faces [self.ndims-1])

        # ALTERNATIVE ALGORITHM:
        # compute facets directly as maximal extremal sets of vertices
        
        # now do edges up to codimension 2 faces
        for k in range(1, self.ndims-1):
            logger.debug(f"Extracting {k}-diml faces...")
            for supports in itertools.combinations(self.faces[-1], self.ndims-k):
                self.faces[k].append(
                    functools.reduce(set.intersection, supports)
                )

            # now eliminate repetitions --- recall sets aren't hashable
            eliminate_repetitions(self.faces[k])
            logger.debug(f"Found up-to-k-diml faces {self.faces[k]}.")

            # remove empty "faces"
            if set() in self.faces[k]:
                logger.debug(f"Removing empty 'face' from list of {k}-faces.")
                self.faces[k].remove(set())

            # faces may contain things of lower dimension
            removals = set()
            for j in range(0,k):
                for i, face in enumerate(self.faces[k]):
                    logger.debug(f"Checking {k}-face {face}...")
                    if face in self.faces[j]:
                        logger.debug(f"Removing face {face} from list of {k}-faces.")
                        removals.add(i)

            for i in sorted(removals, reverse=True):
                self.faces[k].pop(i)


        # OPTIMIZATION: run over intersections kface \cap facet
        # O(ndims*N^2) instead of O(N^ndims) (kind of)
           
    def are_incident(self, i: int, j: int) -> bool:
        '''
        Predicate: two facets with given indices intersect in 
        codimension one.
        '''
        q =  set.intersection(self.faces[-1][i], self.faces[-1][j])
        return q in self.faces[-2]

#    def facet_incidence_graph(self) -> nx.Graph:
#        '''
#        Returns NetworkX Graph object representing incidences between facets.
#        '''
#        incidences = (
#            (i,j) 
#            for i, j in itertools.combinations(range(len(self.faces[-1])), 2)
#            if self.are_incident(i,j)
#        )
#        # ALT ALGO: iterate over codim 2 faces and check inclusions
#        # ALT ALGO: cache these data at time of discovering codim 2 faces
#        return nx.from_edgelist(incidences)
