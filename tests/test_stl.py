from tope.stl import create_stl_walls
from tope.tope import Tope

def test_stl():
    cube = Tope.from_vertices([
        [1,1,1],    [1,1,-1],
        [1,-1,1],   [1,-1,-1],
        [-1,1,1],   [-1,1,-1],
        [-1,-1,1,], [-1,-1,-1]
    ])
    m = create_stl_walls(cube)
    assert m.check()
