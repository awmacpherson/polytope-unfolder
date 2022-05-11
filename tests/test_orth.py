from tope.orth import *
from tope.tope import Tope
from loguru import logger

rng = np.random.default_rng()

ABS_TOL = 1e-6

def test_span():
    A = np.array([[1,-1],[2,-2],[-1,1]])
    A_ = in_own_span(A)
    assert A_.shape[0] == A.shape[0]
    assert A_.shape[1] == 1

def test_get_basis():
    P = Tope.from_vertices([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    F0 = P.faces[-1][0]
    F1 = P.faces[-1][1]
    I = set.intersection(F0,F1)
    assert P.dim == 4

    flag = list(I)[:P.dim-1] 
    I_vertices = P.vertices[flag] # vertices
    logger.debug(f"Compute offset for {I_vertices}.")

    offset = I_vertices.mean(axis=0)
    logger.debug(f"Got offset {offset}.")

    # at this point, rows of (vertices - offset) are linearly dependent
    # overwrite last row with a vertex of F0 \ I
    F0_vertices = I_vertices
    F0_vertices[-1] = P.vertices[F0.difference(I).pop()]

    logger.debug(f"Compute spanning hyperplane for {F0_vertices}.")

    # transpose, add a column of zeros, and reset origin
    flag_basis = np.c_[F0_vertices.T, np.zeros(P.dim)] - np.array([offset]).T

    # now first N-2 columns span I, first N-1 columns span F0, all columns span R^N

    orth_basis, R = np.linalg.qr(flag_basis)

    # signs might be fucked up
    for i in range(len(R)): 
        if R[i,i] < 0:
            orth_basis[:,i] *= -1

    logger.debug(f"Got basis: \n{orth_basis}")

    first = P.vertices[flag[0]] - offset
    logger.debug(f"First vector: {first}")
    unnormalised = orth_basis[:,0]*np.linalg.norm(first)
    logger.debug(f"Unnormalized: {unnormalised}")
    assert np.linalg.norm( unnormalised  - first ) < ABS_TOL

import itertools

def test_rotate():
    P = Tope.from_vertices([[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]])
    #P = Tope([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]])
    F0 = P.faces[-1][0]
    F1 = P.faces[-1][1]
    I = set.intersection(F0,F1)
    assert len(I) == 2

    rotation, offset = rotate_into_hyperplane(P.vertices, F0, F1, I)
    assert rotation.ndim == 2
    assert rotation.shape[0] == rotation.shape[1] == P.dim

    # rotation preserves lengths
    r = rng.normal(size=(100,P.dim))
    assert (np.abs(np.linalg.norm(r @ rotation.T, axis=1) -
            np.linalg.norm(r, axis=1)) < ABS_TOL).all()

    F0_vertices = P.vertices[list(F0)] - offset
    F1_vertices = P.vertices[list(F1)] - offset
    logger.debug(f"Going to rotate \n{F1_vertices}")
    assert np.linalg.matrix_rank(F0_vertices, tol=ABS_TOL) < P.dim
    assert np.linalg.matrix_rank(F1_vertices, tol=ABS_TOL) < P.dim

    F1_vertices_rotated = F1_vertices @ rotation.T

    # check the transformation preserved length
    assert (np.abs(np.linalg.norm(F1_vertices_rotated, axis=1) -
            np.linalg.norm(F1_vertices, axis=1)) < ABS_TOL).all()

    # if it worked, this should be contained in a hyperplane
    flattened = np.concatenate([F0_vertices, F1_vertices_rotated], axis=0)
    logger.debug(f"Received vectors \n{flattened}")
    assert np.linalg.matrix_rank(flattened, tol=ABS_TOL) < P.dim


ROTATOOOR = np.array([[0,-1],[1,0]]) # rotate e_1 -> e_2

def test_rotatooor():
    v = rng.normal(size=2)
    w = rng.normal(size=2)
    w_ = ROTATOOOR @ w
    nrm = np.linalg.norm(v) * np.linalg.norm(w)
    cos = np.dot(v,-w) / nrm
    sin = np.dot(v,-w_) / nrm
    assert abs(cos**2 + sin**2 - 1) < ABS_TOL
