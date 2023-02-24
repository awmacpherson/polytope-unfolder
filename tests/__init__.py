v_3simplex = [[1,0,0],[0,1,0],[0,0,1],[-1,-1,-1]]
v_4simplex = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]
v_24cell = [
    [1, 0, 0, 0], [-1, 0, 0, 0],
    [0, 1, 0, 0], [0, -1, 0, 0],
    [0, 0, 1, 0], [0, 0, -1, 0], 
    [0, 0, 0, 1], [0, 0, 0, -1],
    [1, -1, 0, 0], [-1, 1, 0, 0], 
    [1, 0, -1, 0], [-1, 0, 1, 0], 
    [1, 0, 0, -1], [-1, 0, 0, 1], 
    [0, 1, -1, 0], [0, -1, 1, 0], 
    [0, 1, 0, -1], [0, -1, 0, 1],
    [1, 0, -1, -1], [-1, 0, 1, 1], 
    [0, 1, -1, -1], [0, -1, 1, 1],
    [1, 1, -1, -1], [-1, -1, 1, 1]
]

POLYS_PATH = "data/polys2.json"

from tope.orth import angle_between, ABS_TOL
import numpy as np

def normalize_polygon(A, flip=False):
    A = A - A.mean(axis=0)
    if flip:
        A[:,0] = -A[:,0] # reflect in y-axis
    theta, cos_theta, sin_theta = angle_between(A[0], np.array([1,0]))
    rotator = np.array( [ [ cos_theta, sin_theta ], [ -sin_theta, cos_theta ] ] )
    A_rotated = A @ rotator
    assert np.abs(A_rotated[0,1]) < ABS_TOL
    return A_rotated
