from tope import *
from tope.net import *
POLYS_PATH = "polys.json"
import json
from loguru import logger

with open(POLYS_PATH) as fd:
    polys = json.load(fd)

simplex = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]

vecs = np.array([[1,0,0],[0,2,0],[-1,-1,-1],[3,4,2]])

def test_facet_graph():
    for poly in polys.values():
        P = Tope.from_vertices(poly)
        G = get_facet_graph(P)

        # basic invariants
        #assert G.nodes == set(range(len(P.faces[-1])))

        for i in G.nodes:
            for j in G.children[i]:
                assert G.edge_labels[(i,j)] ==\
                        set.intersection(G.node_labels[i], G.node_labels[j])

        T = G.get_spanning_tree()
        #assert set(P.vindex) == T.nodes
 
        for i in T.nodes:
            for j in T.children[i]:
                assert T.edge_labels[(i,j)] ==\
                        set.intersection(T.node_labels[i], T.node_labels[j])

from numpy.random import default_rng
rng = default_rng()

def test_init():
    P = Tope.from_vertices(simplex)
    G = get_facet_graph(P)
    T = G.get_spanning_tree()
    assert len(T.nodes) == len(G.nodes)
    N = Net(P, T)
    for i, facet in N.facets.items():
        assert facet.shape == (len(P.faces[-1][i]), 4)

def test_unfold():
    P = Tope.from_vertices(simplex)
    G = get_facet_graph(P)
    T = G.get_spanning_tree()
    N = Net(P, T)
    N.unfold()

    vertices = np.concatenate(list(N.facets.values()))
    assert affine_span_dim(vertices) == P.dim - 1

    N = N.in_own_span()
    for _, v in N.facets.items():
        assert v.shape[1] == P.dim - 1

