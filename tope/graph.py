from dataclasses import dataclass
from typing import Any
from loguru import logger

Graph=None

@dataclass
class Graph:
    nodes: set[int]
    node_labels: dict[int, Any]
    edge_labels: dict[tuple[int, int], Any] # include diagonal?
    children: dict[int, set] # indices are elements of self.nodes

    @classmethod
    def empty(cls):
        return cls(set(), {}, {}, {})

    @classmethod
    def from_pairing(cls, nodes: set, pair, node_labels={}):
        # We could have pred return the edge label.
        # not optimised for irreflexive or symmetric predicates
        children = {i: set() for i in nodes} # actually children
        edge_labels = {}
        nodes = set(nodes)
        for node1 in nodes:
            for node2 in nodes:
                if (label := pair(node1, node2)) is not None: # Python 3.8
                    children[node1].add(node2)
                    edge_labels[(node1, node2)] = label
        return cls(nodes, node_labels, edge_labels, children)

#    def iter_edge_labels(self, src):
#        return ((node_labels[src], node_labels[child], edge_labels[(src, child)])\
#                for child in self.children[src])

    def update(self, G: Graph):
        self.nodes.update(G.nodes) # assume disjoint
        self.node_labels.update(G.node_labels)
        self.edge_labels.update(G.edge_labels)
        self.children.update(G.children)

    def join_with(self, G: Graph, at: tuple[int,int], label=None):
        self.update(G)
        self.children[at[0]].add(at[1])
        self.edge_labels[at] = label

    def get_spanning_tree(self, root=None, skip=set()) -> Graph: 
        nodes = set.difference(self.nodes, skip)
        #Check if {nodes} is empty
        if not nodes: 
            logger.debug("Empty!")
            return Graph.empty()
        
        # just pick lowest index for root
        root = min(nodes) if root is None else root
        skip.add(root)

        # init punctual graph
        T = Graph(
            {root}, 
            {root: self.node_labels.get(root)}, 
            {},
            {root: set()}
        )
        T.root = root

        # join children
        for node in self.children[root]:
            if node in skip: continue
            
            T.join_with(
                self.get_spanning_tree(
                    root=node, 
                    skip=skip
                ),
                at=(root, node),
                label=self.edge_labels[(root,node)]
            )
        return T
