from stl import mesh
from .orth import ABS_TOL
import numpy as np

from .net import Net

def create_stl_from_net(N: Net, thickness: float, walls=False) -> mesh.Mesh:

    # If you want the stl with the walls
    if walls:
        return create_stl_walls(*N.facets.values())
   
    # If you want the thick grid
    else:
        return create_stl_grid(N, thickness)

def create_stl_walls(*topes) -> mesh.Mesh:
    # Define triangles from triangulation of the faces
    triangles = [tri for top in topes for tri in top.triangulate()]
    model = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    model.vectors[:,:,:] = triangles
    return model

def create_stl_grid(N: Net, thickness: float) -> mesh.Mesh:
    # Deduplicate vertices
    small_vertices = []
    vertices = [x for x in N.iter_faces_as_arrays(0)]

    for x in vertices:
        eqs = [y for y in small_vertices if (np.abs(x[0]-y[0])<ABS_TOL).all()]
        if len(eqs) == 0:
            small_vertices.append(x)

    # Deduplicate edges
    small_edges = []
    edges = [x for x in N.iter_edges()]

    for x in edges:
        eqs_ord = [y for y in small_edges if (np.abs(x[0]-y[0])<ABS_TOL).all() and (np.abs(x[1]-y[1])<ABS_TOL).all()]
        eqs_rev = [y for y in small_edges if (np.abs(x[1]-y[0])<ABS_TOL).all() and (np.abs(x[0]-y[1])<ABS_TOL).all()]
        if len(eqs_ord) == 0 and len(eqs_rev) == 0:
            small_edges.append(x)

    # Generate one random 3D vector
    rv1 = np.random.rand(3)

    # Store triangles
    triangles = []

    # Define edge prism
    for x in small_edges:
        triangles += edge_prism(x, thickness, rv1)

    # Define vertices shape
    for x in small_vertices:
        x =  x[0]
        triangles+= icosahedron_mesh(x, thickness)

    # Define model
    model = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    model.vectors[:,:,:] = triangles
    return model

def icosahedron_mesh(P:np.array, r:float) -> list:
    # Define the canonical icosahedron vertices
    tau = (1 + np.sqrt(5)) / 2  # Golden ratio
    vertices = np.array([[-1, tau, 0],[1, tau, 0],[-1, -tau, 0],[1, -tau, 0],[0, -1, tau],[0, 1, tau],[0, -1, -tau],
        [0, 1, -tau],[tau, 0, -1],[tau, 0, 1],[-tau, 0, -1],[-tau, 0, 1]])

    # Scale and translate the icosahedron
    scaled_vertices = r * vertices + P

    # Define the 20 faces (each face consists of 3 vertices)
    faces = [[scaled_vertices[0], scaled_vertices[11], scaled_vertices[5]],
        [scaled_vertices[0], scaled_vertices[5], scaled_vertices[1]],
        [scaled_vertices[0], scaled_vertices[1], scaled_vertices[7]],
        [scaled_vertices[0], scaled_vertices[7], scaled_vertices[10]],
        [scaled_vertices[0], scaled_vertices[10], scaled_vertices[11]],
        [scaled_vertices[1], scaled_vertices[5], scaled_vertices[9]],
        [scaled_vertices[5], scaled_vertices[11], scaled_vertices[4]],
        [scaled_vertices[11], scaled_vertices[10], scaled_vertices[2]],
        [scaled_vertices[10], scaled_vertices[7], scaled_vertices[6]],
        [scaled_vertices[7], scaled_vertices[1], scaled_vertices[8]],
        [scaled_vertices[3], scaled_vertices[9], scaled_vertices[4]],
        [scaled_vertices[3], scaled_vertices[4], scaled_vertices[2]],
        [scaled_vertices[3], scaled_vertices[2], scaled_vertices[6]],
        [scaled_vertices[3], scaled_vertices[6], scaled_vertices[8]],
        [scaled_vertices[3], scaled_vertices[8], scaled_vertices[9]],
        [scaled_vertices[4], scaled_vertices[9], scaled_vertices[5]],
        [scaled_vertices[2], scaled_vertices[4], scaled_vertices[11]],
        [scaled_vertices[6], scaled_vertices[2], scaled_vertices[10]],
        [scaled_vertices[8], scaled_vertices[6], scaled_vertices[7]],
        [scaled_vertices[9], scaled_vertices[8], scaled_vertices[1]]]

    return faces

def edge_prism(edge: np.stack, t: float, rv1: np.ndarray) -> list:
    # Define the two vectors defining an edge
    p1 = edge[0]
    p2 = edge[1]

    # Calculate the direction of the edge
    direction_vector = p1 - p2

    # Calculate the cross product to get the normal directions
    n1 = np.cross(direction_vector, rv1)
    n2 = np.cross(direction_vector, n1)

    # Shift vertices in normal directions
    p1_up, p1_down, p1_right, p1_left = p1 + t*n1/np.linalg.norm(n1), p1 - t*n1/np.linalg.norm(n1), p1 + t*n2/np.linalg.norm(n2), p1 - t*n2/np.linalg.norm(n2)
    p2_up, p2_down, p2_right, p2_left = p2 + t*n1/np.linalg.norm(n1), p2 - t*n1/np.linalg.norm(n1), p2 + t*n2/np.linalg.norm(n2), p2 - t*n2/np.linalg.norm(n2)

    # Define triangles needed for the prism
    triangles = [[p1_up, p1_right, p2_up], [p2_up, p2_right, p1_right], [p1_right, p1_down, p2_right], [p2_right, p2_down, p1_down],
                [p1_down, p1_left, p2_down], [p2_down, p2_left, p1_left], [p1_left, p1_up, p2_left], [p2_left, p2_up, p1_up] ]
    return triangles

def stl_dimensions(model: mesh.Mesh):
    # Extract vertices
    vertices = [x for y in model.vectors for x in y]

    # Find the minimum and maximum coordinates in each dimension
    x_min, y_min, z_min = np.min(vertices, axis=0)
    x_max, y_max, z_max = np.max(vertices, axis=0)

    # Calculate the dimensions (numpy.stl dimensions are in mm)
    x_dim_mm = (x_max - x_min) 
    y_dim_mm = (y_max - y_min) 
    z_dim_mm = (z_max - z_min) 

    print("Model dimensions:")
    print(f"X dimension: {x_dim_mm:.2f} mm")
    print(f"Y dimension: {y_dim_mm:.2f} mm")
    print(f"Z dimension: {z_dim_mm:.2f} mm")

def define_scale(N: Net, box_dim: float, t:float) -> float:
    # Extract vertices
    vertices = [x for y in N.iter_edges() for x in y]

    # Find the minimum and maximum coordinates in each dimension
    x_min, y_min, z_min = np.min(vertices, axis=0)
    x_max, y_max, z_max = np.max(vertices, axis=0)

    # Calculate the dimensions (numpy.stl dimensions are in mm)
    x_dim_mm = (x_max - x_min) 
    y_dim_mm = (y_max - y_min) 
    z_dim_mm = (z_max - z_min) 

    # Take the maximum distance
    max_dim = max([x_dim_mm,y_dim_mm,z_dim_mm])

    # Return scale by dividing the desired box dimension by the maximum dimension
    return (box_dim-3*t)/max_dim