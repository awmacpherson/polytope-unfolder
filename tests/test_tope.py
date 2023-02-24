from tope import *
from tope.orth import ABS_TOL, angle_between
from tope.net import get_facet_graph
from tests import normalize_polygon, v_24cell, v_4simplex, v_3simplex, POLYS_PATH


import json
from loguru import logger
from numpy.random import default_rng; rng = default_rng()

def test_meta():
    Delta = Tope.from_vertices(v_24cell)
    Delta.save_index(key = "pasta")
    for k in range(len(Delta.faces)):
        for i in range(len(Delta.faces[k])):
            assert Delta.meta[k][i]["pasta"] == i

    # Test preservation of meta on get_face
    Gamma = Delta.get_face(0, 2)
    for k, k_faces in enumerate(Gamma.faces):
        for i in range(len(k_faces)):
            assert Delta.get_face(Gamma.meta[k][i]["pasta"], k) == Gamma.get_face(i, k)

    # Test apply_to
    Delta.apply_to(lambda x : -x, "pasta")
    for k in range(Delta.dim):
        for i in range(len(Delta.faces[k])):
            assert Delta.meta[k][i]["pasta"] == -i

    # with some entries missing..
    for m in Delta.meta[0]:
        m.clear()
    Delta.apply_to(lambda x : -x, "pasta")
    for k in range(1, Delta.dim):
        for i in range(len(Delta.faces[k])):
            assert Delta.meta[k][i]["pasta"] == i

with open(POLYS_PATH) as fd:
    polys = json.load(fd)

def test_init():
    for idx, thing in polys.items():
        Delta = Tope.from_vertices(thing)

        assert len(Delta.faces) == Delta.dim + 1
        
        # sanity checks
        for v in Delta.faces[0]:
            assert len(v) == 1
        for e in Delta.faces[1]:
            assert len(e) == 2

        for g in Delta.faces[Delta.dim-2]:
            g_in = []
            for f in Delta.faces[Delta.dim-1]:
                
                if g.issubset(f):
                    g_in.append(f)
            assert len(g_in) == 2

    Delta = Tope.from_vertices(v_4simplex)
    assert len(Delta.faces) == 5
    assert len(Delta.faces[1]) == len(Delta.faces[2])

def test_iterators():
    Delta = Tope.from_vertices(v_4simplex)

    # iter_faces(n) returns same iterator as iter(Delta.faces[n])
    for n in range(5):
        for a, b in zip(Delta.iter_faces(n), Delta.faces[n]):
            assert a == b

    # iter_all_faces() returns concatenation of iter_faces(dim=n)
    all_faces = Delta.iter_all_faces()
    all_n_faces = [Delta.iter_faces(dim=n) for n in range(5)]
    assert len(list(all_faces)) == sum((len(l) for l in all_n_faces))

    # enumerate_faces returns ??

    # iter_meta(n) returns same iterator as iter(Delta.meta[n])
    # ...

def test_interface():
    Delta = Tope.from_vertices(v_4simplex)
    for i in range(5):
        for j in range(5):
            assert (Delta.interface(i,j) is None) == (i==j)

def test_in_own_span():
    Delta = Tope.from_vertices(v_4simplex)
    for i in range(5):
        Delta.meta[0][i]["butter"] = rng.random(size=4)
    F = Delta.get_facet(0, meta_keys = ["butter"])
    
    assert F.dim == 3
    # ordering of vertices preserved
    assert sorted(F.faces[0]) == F.faces[0]

    for i in range(4):
        assert len(F.meta[0][i]["butter"]) == 3

def test_orientations():
    P = Tope.from_vertices(v_4simplex)
    P.save_index()
    for i in range(4):
        for j in range(4):
            if i == j: continue
            # get index in list of codim 2 faces of intersection
            I = set.intersection(P.faces[P.dim-1][i], P.faces[P.dim-1][j])
            k = P.faces[P.dim-2].index(I)

            Fi = P.get_facet(i) # triangle
            idx_k_in_i = [meta["index"] for meta in Fi.meta[Fi.dim-1]].index(k)
            or_k_in_i = np.linalg.det(Fi.vertices[list(Fi.faces[Fi.dim-1][idx_k_in_i])])
            
            Fj = P.get_facet(j)
            idx_k_in_j = [meta["index"] for meta in Fj.meta[Fj.dim-1]].index(k)
            or_k_in_j = np.linalg.det(Fj.vertices[list(Fj.faces[Fj.dim-1][idx_k_in_j])])

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
    P.save_index()

    for i, j in get_facet_graph(P).edge_labels:
        # get index of intersection
        I = set.intersection(P.faces[P.dim-1][i], P.faces[P.dim-1][j])
        k = P.faces[P.dim-2].index(I)

        Fi = P.get_facet(i) # 3-cell
        idx_k_in_i = [meta["index"] for meta in Fi.meta[Fi.dim-1]].index(k)
        Gik = Fi.get_facet(idx_k_in_i).vertices
        Gik = normalize_polygon(Gik)

        Fj = P.get_facet(j)
        idx_k_in_j = [meta["index"] for meta in Fj.meta[Fj.dim-1]].index(k)
        Gjk = Fj.get_facet(idx_k_in_j).vertices
        Gjk = normalize_polygon(Gjk, flip=True)


        assert (Gik - Gjk < ABS_TOL).all()

        Gk = P.get_face(k, -2).in_own_span().vertices # orientation not defined
        assert ((normalize_polygon(Gk) - Gik) < ABS_TOL).all() or \
                ((normalize_polygon(Gk, True) - Gik) < ABS_TOL).all()

        if i > 7: break # speed things up a bit

def test_cut_faces():
    P = Tope.from_vertices(v_3simplex)

    # with empty list
    P.cut_2faces_with_hyperplanes([])

    hyperplanes = [(rng.normal(size=3), np.array([a,a,a])) for a in np.arange(-1,1,.2)]
    P.cut_2faces_with_hyperplanes(hyperplanes)
    
    for k in range(len(P.faces[2])):
        size = P.meta[2][k]["cuts"].shape
        assert size[0] > 0
        assert size[1] == 2
        assert size[2] == P.dim
