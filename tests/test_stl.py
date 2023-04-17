from tope.stl import create_stl
from tope.tope import Tope

def test_stl():
    cube = Tope.from_vertices([
        [1,1,1],    [1,1,-1],
        [1,-1,1],   [1,-1,-1],
        [-1,1,1],   [-1,1,-1],
        [-1,-1,1,], [-1,-1,-1]
    ])
    m = create_stl(cube)
    assert m.check()
