from tope import *
from tope.orth import ABS_TOL, angle_between
from tope.net import get_facet_graph
from tests import normalize_polygon, v_24cell, v_4simplex

POLYS_PATH = "polys.json"
import json
from loguru import logger


with open(POLYS_PATH) as fd:
    polys = json.load(fd)

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

    Delta = Tope.from_vertices(v_4simplex)
    assert len(Delta.faces) == 4
    assert len(Delta.faces[1]) == len(Delta.faces[2])

def test_interface():
    Delta = Tope.from_vertices(v_4simplex)
    for i in range(5):
        for j in range(5):
            assert (Delta.interface(i,j) is None) == (i==j)

def test_in_own_span():
    Delta = Tope.from_vertices(v_4simplex)
    F = Delta.get_facet(0)
    #F = F.in_own_span()
    assert F.dim == 3
    # ordering of vertices preserved
    assert sorted(F.faces[0]) == F.faces[0]

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

def normalize_polygon(A, flip=False):
    if flip:
        A[:,0] = -A[:,0] # reflect in y-axis
    theta, cos_theta, sin_theta = angle_between(A[0], np.array([1,0]))
    rotator = np.array( [ [ cos_theta, sin_theta ], [ -sin_theta, cos_theta ] ] )
    A_rotated = A @ rotator
    assert np.abs(A_rotated[0,1]) < ABS_TOL
    return A_rotated

def test_similarity():
    "Test that common 2-faces of adjacent 3-faces are similar up to a flip."
    P = Tope.from_vertices(v_24cell)
    for i, j in get_facet_graph(P).edge_labels:
        # get index of intersection
        I = set.intersection(P.faces[-1][i], P.faces[-1][j])
        k = P.faces[-2].index(I)

        Fi = P.get_facet(i) # 3-cell
        idx_k_in_i = Fi.labels[-1].index(k)
        Gik = Fi.get_facet(idx_k_in_i).vertices
        Gik = normalize_polygon(Gik)

        Fj = P.get_facet(j)
        idx_k_in_j = Fj.labels[-1].index(k)
        Gjk = Fj.get_facet(idx_k_in_j).vertices
        Gjk = normalize_polygon(Gjk, flip=True)


        assert (Gik - Gjk < ABS_TOL).all()

        Gk = P.get_face(k, -2).in_own_span().vertices # orientation not defined
        assert ((normalize_polygon(Gk) - Gik) < ABS_TOL).all() or \
                ((normalize_polygon(Gk, True) - Gik) < ABS_TOL).all()

        if i > 7: break # speed things up a bit
