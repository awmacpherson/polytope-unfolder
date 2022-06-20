from tope import *
POLYS_PATH = "polys.json"
import json
from loguru import logger


with open(POLYS_PATH) as fd:
    polys = json.load(fd)

simplex = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]

def test_init():
    for idx, thing in polys.items():
        logger.warning(f"Check poly {idx}...")
        Delta = Tope.from_vertices(thing)

        assert len(Delta.faces) == Delta.dim
        
        # sanity checks
        for v in Delta.faces[0]:
            assert len(v) == 1
        for e in Delta.faces[1]:
            assert len(e) == 2

        for g in Delta.faces[-2]:
            g_in = []
            for f in Delta.faces[-1]:
                
                if g.issubset(f):
                    g_in.append(f)
            assert len(g_in) == 2

    Delta = Tope.from_vertices(simplex)
    assert len(Delta.faces) == 4
    assert len(Delta.faces[1]) == len(Delta.faces[2])

def test_interface():
    Delta = Tope.from_vertices(simplex)
    for i in range(5):
        for j in range(5):
            assert (Delta.interface(i,j) is None) == (i==j)

def test_in_own_span():
    Delta = Tope.from_vertices(simplex)
    F = Delta.get_facet(0)
    #F = F.in_own_span()
    assert F.dim == 3

tetra = [[1,0,0], [0,1,0], [0,0,1], [-1,-1,-1]]

def test_orientations():
    P = Tope.from_vertices(tetra)
    for i in range(4):
        for j in range(4):
            if i == j: continue
            # get index of intersection
            I = set.intersection(P.faces[-1][i], P.faces[-1][j])
            k = P.faces[-2].index(I)

            Fi = P.get_facet(i) # triangle
            idx_k_in_i = Fi.labels[-1].index(k)
            or_k_in_i = np.linalg.det(Fi.vertices[list(Fi.faces[-1][idx_k_in_i])])

            Fj = P.get_facet(j)
            idx_k_in_j = Fj.labels[-1].index(k)
            or_k_in_j = np.linalg.det(Fj.vertices[list(Fj.faces[-1][idx_k_in_j])])

            assert or_k_in_i * or_k_in_j < 0
