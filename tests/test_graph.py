from tope.graph import Graph

edges_1 = [(0,1),(1,2),(2,4), (1,4)]
edges_2 = [(5,6), (6,5)]

def pairing(d, symmetrize=True):
    if symmetrize:
        def pair(i,j):
            return i+j if (i,j) in d or (j,i) in d else None
    else:
        def pair(i,j):
            return i+j if (i,j) in d else None
    return pair

def test_init():
    G = Graph.from_pairing(range(5), pairing(edges_1))
    for e in edges_1:
        assert e[1] in G.children[e[0]]
        assert e[0] in G.children[e[1]]

    G = Graph.empty()
    assert len(G.nodes) == len(G.node_labels) == len(G.edge_labels) ==\
            len(G.children) == 0

def test_join_with():
    G = Graph.from_pairing(range(5), pairing(edges_1))
    H = Graph.from_pairing(range(5,7), pairing(edges_2))
    G.join_with(H, at=(3,5), label="WUT")
    assert len(G.children[3]) == 1
    assert G.edge_labels[(3,5)] == "WUT"

def test_width_first_spanning_tree():
    G = Graph.from_pairing(range(0,5), pairing(edges_1))
    G.join_with(
        Graph.from_pairing(range(5,7), pairing(edges_2)),
        at=(2,5)
    )

    T = G.width_first_spanning_tree(root=0)

    assert hasattr(T, 'root')
    assert T.nodes == set(range(7)).difference({3})
    for i in T.nodes:
        for j in T.nodes:
            assert not (i in T.children[j] and j in T.children[i])


    def collect(start, into):
        for i in T.children[start]:
            into.append(i)
            collect(i, into)

    l = [0]
    collect(0, l)
    assert len(l) == len(T.nodes) 
    # every node other than root appears exactly once as a child
    # TODO: test unique path from root to child

    T = Graph.empty().get_spanning_tree()
    assert T == Graph.empty()

def test_spanning_tree():
    G = Graph.from_pairing(range(0,5), pairing(edges_1))
    G.join_with(
        Graph.from_pairing(range(5,7), pairing(edges_2)),
        at=(2,5)
    )

    T = G.get_spanning_tree(root=0)

    assert hasattr(T, 'root')
    assert T.nodes == set(range(7)).difference({3})
    for i in T.nodes:
        for j in T.nodes:
            assert not (i in T.children[j] and j in T.children[i])


    def collect(start, into):
        for i in T.children[start]:
            into.append(i)
            collect(i, into)

    l = [0]
    collect(0, l)
    assert len(l) == len(T.nodes) 
    # every node other than root appears exactly once as a child
    # TODO: test unique path from root to child

    T = Graph.empty().get_spanning_tree()
    assert T == Graph.empty()

def test_iter_from():
    G = Graph.from_pairing(range(0,5), pairing(edges_1, symmetrize=False))
    l = list(G.iter_from(1))
    assert len(l) == 4
    for i in l:
        assert type(i) == int
    assert l[0] == 1
    assert l[3] == 4
