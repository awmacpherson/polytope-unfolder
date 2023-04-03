import numpy as np
from typing import Union
from loguru import logger
import itertools

ABS_TOL = 1e-6
ORTH_RNG = np.random.default_rng()

intersection_err = "Got more than 2 endpoints when intersecting " +\
        "{} with {}.\nIntersections: {}\nI need help sorting this out!"

# Projections

def perspective_project(v: np.ndarray, offset: float):
    """
    Convention: project into hyperplane (v[0]=0) translated by offset.
    - v: float[M,N]
    - offset: float
    """
    return v[:,...,1:] / (v[:,...,:1] - offset)

# Random orthogonal matrices

def random_orth(N: int, rng: np.random._generator.Generator = ORTH_RNG) -> np.ndarray:
    """
    Uniform random element of O(N) with positive diagonal entries sampled using Haar measure.
    """
    Q,R = np.linalg.qr(rng.normal(size=(N,N))) # project GL(N) -> O(N)
    return Q @ np.diag(np.sign(np.diag(R)))    # fix signs

# Intersections

def intersect_polygon_with_hyperplane(polygon, hyperplane) -> np.ndarray: # [2]float
    intersections = []
    for edge in polygon.faces[1]:
        intersections.extend(
                intersect_line_segment_with_hyperplane(
                    polygon.vertices[list(edge)], hyperplane
                    )
                )

    if len(intersections) > 2: 
        logger.debug("Found more than 2 intersections. Attempting to remove repeated instances...")
        # There could be an intersection at a vertex that was counted twice.
        # Build up list of redundant intersections.
        removals = set()
        for u, v in itertools.combinations(enumerate(intersections), r=2):
            i, u = u
            j, v = v
            if (np.abs(u-v) < ABS_TOL).all():
                removals.add(j)

        # Pop removals in reverse order
        for r in sorted(removals, reverse=True):
            intersections.pop(r)

    if len(intersections) < 2:
        return None
    
    if len(intersections) == 2:
        return np.array(intersections) # same as np.stack

    if np.abs((polygon.vertices - hyperplane[1]) @ hyperplane[0]).all() < ABS_TOL:
        # entire polygon is contained in the hyperplane.
        return None

    # If we reached this point, something went unrecoverably wrong.
    raise ValueError(intersection_err.format(hyperplane, polygon, intersections))

def intersect_line_segment_with_hyperplane(I, H) -> list:
    """
    Finds point of intersection, if any, of I = (startpt, endpt) with H = (kernel, offset).
    Returns:
    - [] if no intersection;
    - [p] if there is an intersection at point p;
    - list(I) if I is contained in H.
    """
    A, offset = H
    u, v = I[0] - offset, I[1] - offset # homogenize

    pos = A @ v
    vel = A @ (u - v) # in dual line to {A = 0}
    if vel == 0:
        return [] # list(I) if pos == 0 else []
    t = -pos / vel
    if t > -ABS_TOL and t < 1 + ABS_TOL:
        return [t * I[0] + (1-t) * I[1]]
    return []



def intersect_set_with_affine_subspace(vertices: np.ndarray, A, b) -> set:
    return set( np.arange(len(vertices)) [np.abs(A @ vertices.T - b) < ABS_TOL] )


# Reencoding to a subspace

def affine_span_dim(A: np.ndarray) -> int:
    return np.linalg.matrix_rank(A - A.mean(axis=0), tol=ABS_TOL)

def linear_span_dim(A: np.ndarray) -> int:
    "Return the dimension of the linear subspace spanned by some points."
    return np.linalg.matrix_rank(A, tol=ABS_TOL)

    S = np.linalg.svd(A, compute_uv=False)
    # these assertions all pass:
    # assert S.ndim == 1
    # assert type(S) == np.ndarray
    # assert S.shape[0] == min(A.shape)
    return len(S[np.abs(S)>ABS_TOL])

def in_own_span(A, orientation=None):
    """
    Return row vectors of A expressed in orthonormal basis for their linear span.
    Also return said basis; basis.T is then a map from the ambient space to the subspace.
    If set, orientation is a stack of vectors complementary to A.
    """
    # some issue with complex values?
    U, S, V = np.linalg.svd(A)
    relevant_indices = np.ndarray(len(V), dtype=bool) # init False
    relevant_indices[:len(S)] = np.abs(S)>0.1

    basis = V[relevant_indices]
    logger.debug(f"Found basis of length {len(basis)}. Reencoding in codimension {basis.shape[1]-len(basis)}.")
    if orientation is not None and np.linalg.det(np.c_[basis.T, orientation]) < 0:
        basis[0] = -basis[0] # reverse first basis vector
        
    return (A @ basis.T), basis

# DEPRECATED

def extend_to_orthonormal_basis(A: np.ndarray, orientation = None) -> np.ndarray:
    """
    Return stacked orthonormal matrices of size A.shape[-1] whose first
    A.shape[-2] rows are the rows of A. If orientation is not none, it should
    be 1 or -1 according to the desired orientation of the returned matrix.
    """
    _, _, V = np.linalg.svd(A, full_matrices=True)
    if orientation is not None and orientation * np.linalg.det(V) < 0:
        if len(A) == len(V) - 1:
            V[-1] = -V[-1] # O(N)
        else: # O(1)
            tmp = V[-1]
            V[-1] = V[-2]
            V[-2] = tmp


    V[:len(A)] = A
    return V

def find_basis_in(v: np.ndarray) -> np.ndarray:
    """
    Returns a square submatrix of v that has the same span as v.
    """
    N = linear_span_dim(v)
    span_dim = 0
    indices = [] # populate this with indices at which the dimension of the
                # linear span jumps
    for i in range(len(v)):
        prev_dim = span_dim
        span_dim = linear_span_dim(v[:i+1])
        logger.debug(f"Span dimension of first {i+1} vectors: {span_dim}.")
        if span_dim == prev_dim + 1:
            logger.debug(f"Appending jump index {i}.")
            indices.append(i)
        if span_dim == N:
            return v[indices]
    raise AssertionError(f"We shouldn't have reached here! Found indices {indices}.")

# /DEPRECATED

def find_basis_for_row_space(A: np.ndarray) -> np.ndarray:
    """
    Return orthonormal basis for span of row space of A.
    """
    q, r = np.linalg.qr(A.T)
    #print(r)
    mask = (np.abs(r) > ABS_TOL).any(axis=1)
    return q.T[mask]

# Rotators

# DEPRECATED

def angle_between(v1: np.ndarray, v2: np.ndarray, complement=False) -> (float, float, float):
    "Returns the angle in degrees between two vectors along with its cosine and sine."
    nrm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if nrm < ABS_TOL: 
        logger.warning(f"Norm is small, angle calculation may be unreliable.")
    cos_theta = np.dot(v1, v2) / nrm
    sin_theta = np.sqrt(1 - cos_theta*cos_theta) # can't be negative!
    
    reflex = np.linalg.det(np.c_[v1,v2]) < 0

    if complement:
        cos_theta = -cos_theta
        if not reflex:
            sin_theta = -sin_theta
    elif reflex:
        sin_theta = -sin_theta

    theta = np.arccos(cos_theta)*180/np.pi
    return theta, cos_theta, sin_theta

def rotation_matrix(cos, sin, N: int) -> np.ndarray:
    """
    Return matrix of clockwise rotation by angle with given sine and cosine
    around axis spanned by first N-2 basis vectors.
    """
    # clockwise rotation by theta (rotation into F0 plane with opposite sign)

    A = np.eye(N)
    A[-2,-2:] = [cos, -sin]
    A[-1,-2:] = [sin, cos]
    return A

# /DEPRECATED

def fold_matrix_2d(v: np.ndarray) -> np.ndarray:
    """
    Matrix of rotation in the plane from [1,0].T to v. For use on row vectors.
    """
    v_n = v / np.linalg.norm(v)
    return np.array([
        [v_n[0], v_n[1]],
        [-v_n[1], v_n[0]]
    ])

def fold_matrix(
        axis: np.ndarray,
        to_v: np.ndarray,
        from_v: np.ndarray
    ):
    """
    Return matrix of rotaion around codimension 2 space with orthonormal basis 
    axis that sends the hyperplane spanned by axis and from_v into the
    hyperplane spanned by axis and to_v. The rotation is "towards" to_v
    if reverse=False, otherwise it is "away from" to_v.
    """
    N = axis.shape[-1]

    # Construct row basis for flag <axis, from_v, to_v>
    basis = np.r_[axis, np.expand_dims(from_v, 0), np.expand_dims(to_v,0)]
    #assert basis.shape[0] == basis.shape[1] == N

    # Begin Gram-Schmidt process for last two vectors
    basis[-2] -= axis.T @ np.dot(axis, from_v)   # project out <axis>
    basis[-1] -= axis.T @ np.dot(axis, to_v)     # project out <axis>

    # At this point, last two basis elements may fail to be orthogonal
    # save direction vector from to_v
    v = basis[-1].copy()

    # Now complete Gram-Schmidt process
    basis[-2] /= np.linalg.norm(basis[-2])                # normalize from_v
    basis[-1] -= basis[-2] * np.dot(basis[-2], basis[-1]) # project out <from_v>
    basis[-1] /= np.linalg.norm(basis[-1])                # normalize to_v

    v = v @ basis.T # encode in row basis
    assert (np.abs(basis @ basis.T - np.eye(N)) < ABS_TOL).all() # basis is orthonormal
    assert (np.abs(v[:-2])<1e-6).all() # passes
    #logger.debug(f"Found basis: \n{basis}")
    #logger.debug(f"Found direction: {v[-2:]}")

    R = np.eye(N)
    R[-2:,-2:] = fold_matrix_2d(-v[-2:]) # right action on row space

    # encode - rotate - decode
    return basis.T @ R @ basis

def fold(x, into, around) -> np.ndarray:
    offset = around.mean(axis=0)
    into_v = into.mean(axis=0) - offset
    x_l = x - offset
    x_v = x_l.mean(axis=0)

    axis = find_basis_for_row_space(around - offset)
    M = fold_matrix(axis, into_v, x_v)
    return (x_l @ M) + offset

def fold_into_hyperplane(vertices, F0, F1) -> np.ndarray:
    x = vertices[sorted(F1)]
    into = vertices[sorted(F0)]
    I = set.intersection(F0,F1)
    around = vertices[sorted(I)]

    offset = around.mean(axis=0)
    into_v = into.mean(axis=0) - offset
    x_l = x - offset
    x_v = x_l.mean(axis=0)

    axis = find_basis_for_row_space(around - offset)
    M = fold_matrix(axis, into_v, x_v)
    return M, offset

# DEPRECATE

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
