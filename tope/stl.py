from stl import mesh
import numpy as np

from .net import Net

def create_stl_from_net(N: Net) -> mesh.Mesh:
    return create_stl(N.facets.values())

def create_stl(*topes) -> mesh.Mesh:
    triangles = [tri for top in topes for tri in top.triangulate()]
    model = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    model.vectors[:,:,:] = triangles
    return model
