from tope.orth import *
from tope.tope import Tope
from loguru import logger
from tests import v_3simplex, normalize_polygon

rng = np.random.default_rng()

ABS_TOL = 1e-6

def test_intersect_line_segment_with_hyperplane():
    hyperplane = (np.array([1,0,0]), np.array([4,3,3]))
    
    # property test (fuzz me)
    seg = (rng.normal(4, size=3), rng.normal(4, 7, size=3))
    for p in intersect_line_segment_with_hyperplane(seg, hyperplane):
        assert np.abs(p[0] - 4) < ABS_TOL

    # no intersection
    seg[0][0] = 3
    seg[1][0] = 3.6
    i = intersect_line_segment_with_hyperplane(seg, hyperplane)
    assert i == []

    # one intersection
    seg[1][0] = 5
    i = intersect_line_segment_with_hyperplane(seg, hyperplane)
    assert len(i) == 1
    assert np.abs(i[0][0] - 4) < ABS_TOL

    # containment --- ignore
    seg[0][0] = 4
    seg[1][0] = 4
    i = intersect_line_segment_with_hyperplane(seg, hyperplane)
    assert len(i) == 0

def test_intersect_polygon_with_hyperplane():
    P = Tope.from_vertices(v_3simplex)
    H = (rng.normal(size=3), np.zeros(3))

    has_intersection = []
    for i in range(4):
        intersections = intersect_polygon_with_hyperplane(P.get_face(i), H)
        assert intersections is None or len(intersections) == 2
        if intersections is not None:
            has_intersection.extend(intersections)
    assert has_intersection

def test_intersect_set_with_affine_subspace():
    verts = np.array([[-1,0],[0,2],[3,3]])
    A = np.array([2,-1])
    b = -2
    assert intersect_set_with_affine_subspace(verts, A, b) == {0, 1}

def test_affine_dim():
    A = np.array([[-1,0],[0,2],[-1.5,-1]])
    assert affine_span_dim(A) == 1

def test_linear_dim():
    A = np.array([[1,-1],[2,-2],[-1,1]])
    assert linear_span_dim(A) == 1
    assert linear_span_dim(np.array([[0,0]])) == 0

def test_span():
    A = np.array([[1,-1],[2,-2],[-1,1]])
    A_, basis = in_own_span(A)
    assert A_.shape[0] == A.shape[0]
    assert A_.shape[1] == 1
    assert basis.shape[0] == 1
    assert basis.shape[1] == A.shape[1]

def test_orientation():
    A = np.array([[1,-1],[2,-2],[-1,1]])
    o = np.array([1,1])
    A_, basis = in_own_span(A, orientation=o)
    assert A_[0] > 0

import itertools

def test_rotate():
    P = Tope.from_vertices(v_3simplex)
    F0 = P.faces[P.dim-1][0]
    F1 = P.faces[P.dim-1][1]

    rotation, offset = rotate_into_hyperplane(P.vertices, F0, F1)
    assert rotation.ndim == 2
    assert rotation.shape[0] == rotation.shape[1] == P.dim

    # rotation preserves lengths
    r = rng.normal(size=(100,P.dim))
    assert (np.abs(np.linalg.norm(r @ rotation.T, axis=1) -
            np.linalg.norm(r, axis=1)) < ABS_TOL).all()

    F0_vertices = P.vertices[sorted(F0)] - offset
    F1_vertices = P.vertices[sorted(F1)] - offset
    logger.debug(f"Going to rotate \n{F1_vertices}")
    assert np.linalg.matrix_rank(F0_vertices, tol=ABS_TOL) < P.dim
    assert np.linalg.matrix_rank(F1_vertices, tol=ABS_TOL) < P.dim

    F1_vertices_rotated = F1_vertices @ rotation

    # check the transformation preserved length
    assert (np.abs(np.linalg.norm(F1_vertices_rotated, axis=1) -
            np.linalg.norm(F1_vertices, axis=1)) < ABS_TOL).all()

    # if it worked, this should be contained in a hyperplane
    flattened = np.concatenate([F0_vertices, F1_vertices_rotated], axis=0)
    logger.debug(f"Received vectors \n{flattened}")
    assert np.linalg.matrix_rank(flattened, tol=ABS_TOL) < P.dim

