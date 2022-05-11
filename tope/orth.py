import numpy as np
from loguru import logger

ABS_TOL = 1e-6

def in_own_span(A):
    "Return row vectors of A expressed in orthonormal basis for their linear span."
    # some issue with complex values?
    U, S, V = np.linalg.svd(A)
    assert A.shape[1] == len(S)
    return (A @ np.linalg.inv(V))[:,np.abs(S)>0.1]

def rotate_into_hyperplane(vertices, F0, F1, I):
    N = vertices.shape[1]
    I_vertices = vertices[list(I)[:N-1]] # vertices
    offset = I_vertices.mean(axis=0)

    # at this point, rows of (vertices - offset) are linearly dependent
    # overwrite last row with a vertex of F0 \ I
    F0_vertices = I_vertices
    F0_vertices[-1] = vertices[F0.difference(I).pop()]

    F1_vertices = vertices[list(F1)] 
    F1_centre = F1_vertices.mean(axis=0) - offset

    # transpose, add a column of zeros, and reset origin
    flag_basis = np.c_[F0_vertices.T, np.zeros(N)] - np.array([offset]).T

    # now first N-2 columns span I, first N-1 columns span F0, all columns span R^N
    orth_basis, R = np.linalg.qr(flag_basis)
    orth_basis_i = np.linalg.inv(orth_basis) # CHECK ERROR

    if np.linalg.norm(orth_basis @ orth_basis_i - np.eye(N)) > ABS_TOL:
        logger.warning("High error in inversion.")

    # orientation might have changed
    # only care about last two coordinates: expect
    # R[-2,-2] > 0, R[-2,-1] < 0, R[-1,-1] > 0
#    assert R[-2,-1] < 0
#    for i in range(len(R)): 
#        if R[i,i] < 0:
#            orth_basis[:,i] *= -1

    
    F0_centre = F0_vertices.mean(axis=0) - offset # not really the centre but doesn't matter

    # orthogonal projection onto 2-plane of rotation
    F0_centre_Q = (orth_basis_i @ F0_centre)[-2:]
    F1_centre_Q = (orth_basis_i @ F1_centre)[-2:]
   
    assert np.abs(F0_centre_Q[1]) < ABS_TOL # should always be zero
    #if F0_centre_Q[0] < 0: F0_centre_Q *= -1
    #if F1_centre_Q[1] < 0: F1_centre_Q += -1
    flip = np.linalg.det(np.stack([F0_centre_Q, F1_centre_Q])) > 0

    if np.linalg.norm(F0_centre_Q) < 0.1 or np.linalg.norm(F1_centre_Q) < 0.1:
        logger.warning("High error in projection.")

    nrm = np.linalg.norm( F1_centre_Q ) * np.linalg.norm( F0_centre_Q )
    if nrm < 0.1: logger.warning(f"Norm is small.")
    cos_theta = np.dot( F1_centre_Q , -F0_centre_Q ) / nrm # CHECK ERROR
    logger.debug(f"Rotating by {np.arccos(cos_theta)*180/np.pi:.2f} degrees...")
    sin_theta = np.sqrt( 1 - cos_theta*cos_theta )


#    assert sin_theta > 0

    # clockwise rotation by theta (rotation into F0 plane with opposite sign)
    rotator_2d = np.array( [ [ cos_theta, sin_theta ], [ -sin_theta, cos_theta ] ] )
    if flip: rotator_2d = rotator_2d.T
#    logger.debug(f"Rotation matrix: \n{rotator_2d}")
    rotator = np.eye(N)
    rotator[-2:,-2:] = rotator_2d

    # rotation must be carried out in new basis
    return orth_basis @ rotator @ orth_basis_i, offset
