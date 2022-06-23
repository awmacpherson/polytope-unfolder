import numpy as np
from typing import Union
from loguru import logger

ABS_TOL = 1e-6

def intersect_set_with_affine_subspace(vertices: np.ndarray, A, b) -> set:
    return set( np.arange(len(vertices)) [np.abs(A @ vertices.T - b) < ABS_TOL] )

def affine_span_dim(A: np.ndarray) -> int:
    return linear_span_dim(A - A.mean(axis=0))

def linear_span_dim(A: np.ndarray) -> int:
    "Return the dimension of the linear subspace spanned by some points."
    U, S, V = np.linalg.svd(A)
    # these assertions all pass:
    # assert S.ndim == 1
    # assert type(S) == np.ndarray
    # assert S.shape[0] == min(A.shape)
    return len(S[np.abs(S)>ABS_TOL])

def in_own_span(A, orientation=None):
    """
    Return row vectors of A expressed in orthonormal basis for their linear span.
    If set, orientation is a stack of vectors complementary to A.
    """
    # some issue with complex values?
    U, S, V = np.linalg.svd(A)
    relevant_indices = np.ndarray(len(V), dtype=bool) # init False
    relevant_indices[:len(S)] = np.abs(S)>0.1

    basis = V[relevant_indices]
    if orientation is not None and np.linalg.det(np.c_[basis.T, orientation]) < 0:
        V[0] = -V[0] # reverse first basis vector
        
    return (A @ V.T)[:,relevant_indices], V[relevant_indices]

def angle_between(v1: np.ndarray, v2: np.ndarray) -> (float, float, float):
    "Returns the angle in degrees between two vectors along with its cosine and sine."
    nrm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if nrm < ABS_TOL: 
        logger.warning(f"Norm is small, angle calculation may be unreliable.")
    cos_theta = np.dot(v1, v2) / nrm
    sin_theta = np.sqrt(1 - cos_theta*cos_theta) # can't be negative!
    if np.linalg.det(np.c_[v1,v2]) < 0:
        sin_theta = -sin_theta
    theta = np.arccos(cos_theta)*180/np.pi

    return theta, cos_theta, sin_theta

def rotate_into_hyperplane(
        vertices: np.ndarray, 
        F0: set[int], 
        F1: set[int]
    ):
    """
    Rotate the set of vertices of a facet into the hyperplane spanned by
    another facet around the axis spanned by their intersection.

    Arguments:
    - vertices. Array of vertices of top-dimensional cell.
    - F0. Set of indices of vertices of the *reference* facet (into whose
    hyperplane the configuration will be rotated).
    - F1. Set of indices of vertices of the facet to be rotated.
    - I = set.intersection(F0, F1)
    """

    N = vertices.shape[1]                   # ambient dimension
    I = set.intersection(F0, F1)

    # Construct an affine basis of R^N with the following properties;
    # 1. First N-2 vectors in affine span of I.
    # 2. First N-1 vectors in affine span of F0.
    # We assume the origin isn't near any vertices (true for reflexive).
    flag_basis = np.zeros((N,N))
    flag_basis[:N-2] = vertices[list(I)][:N-2] # ALWAYS WORKS BECAUSE N < 5
    flag_basis[-2] = vertices[F0.difference(I).pop()]

    ref_pt = offset = vertices[list(I)].mean(axis=0)        # reference point

    # Now apply Gram-Schmidt to get an orthogonal basis
    orth_basis, _   = np.linalg.qr((flag_basis - ref_pt).T)  # Gram-Schmidt decomposition
    orth_basis_i    = orth_basis.T              # Transpose = inverse
    assert (orth_basis_i @ orth_basis - np.eye(N) < ABS_TOL).all()

    # Pick reference points in the interior of each face and
    # orthogonal project onto invariant (N-2)-plane of rotation
    F0_ref_pt   = flag_basis[:N-1].mean(axis=0)     - ref_pt
    F0_centre_Q = (orth_basis_i @ F0_ref_pt)[-2:]
    F1_centre   = vertices[list(F1)].mean(axis=0)   - ref_pt
    F1_centre_Q = (orth_basis_i @ F1_centre)[-2:]
   
    #assert np.abs(F0_centre_Q[1]) < ABS_TOL # should always be zero
    #if np.linalg.norm(F0_centre_Q) < 0.1 or np.linalg.norm(F1_centre_Q) < 0.1:
    #    logger.warning("High error in projection.")

    # Now we construct the rotation...
    theta, cos_theta, sin_theta = angle_between(F1_centre_Q, -F0_centre_Q)
    logger.debug(f"Rotating by {theta:.2f} degrees...")

    # clockwise rotation by theta (rotation into F0 plane with opposite sign)
    rotator_2d = np.array( [ [ cos_theta, sin_theta ], [ -sin_theta, cos_theta ] ] )

    # Now put it together to NxN matrix
    rotator = np.zeros((N,N))
    rotator[:-2,:-2] = np.eye(N-2)
    rotator[-2:,-2:] = rotator_2d

    # rotation must be carried out in new basis
    return orth_basis @ rotator @ orth_basis_i, offset
