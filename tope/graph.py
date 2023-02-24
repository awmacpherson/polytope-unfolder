from dataclasses import dataclass
from typing import Any
from loguru import logger
from numpy.random import default_rng
GRAPH_RNG = default_rng(24)

Graph=None

@dataclass
class Graph:
    """
    Simple directed graph type with labelled edges and nodes.
    Nodes are a set of elements that are assumed sufficiently unique in the
    containing context to use unions for the purposes of constructing joins.
    """
    nodes: set[int]
    node_labels: dict[int, Any]
    edge_labels: dict[tuple[int, int], Any] # include diagonal?
    children: dict[int, set[int]] # indices are elements of self.nodes

    @classmethod
    def empty(cls) -> Graph:
        return cls(set(), {}, {}, {})

    @classmethod
    def punctual(cls, x, label=None) -> Graph:
        return cls({x}, {x: label}, {}, {x: set()})

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

    def _grow_tree(self, T: Graph, leaves=[]) -> list[int]:
        """
        Iterate over leaves, adding one level of new nodes. Assume that leaves
        comprises all leaves of T; all other nodes of T belong to skip.
        """
        new_leaves = []
        logger.debug(f"Doing the leaves: {leaves}")
        for l in leaves:
            for n in self.children[l]:
                if n in T.nodes: continue
                new_leaves.append(n)
                N = Graph(
                        {n}, 
                        {n: self.node_labels.get(n)}, 
                        {},
                        {n: set()}
                        )
                T.join_with(N, at=(l,n), label=self.edge_labels[(l,n)])
        return new_leaves

    def width_first_spanning_tree(self, root=None) -> Graph:
        if root is None:
            root = GRAPH_RNG.choice(list(self.nodes))
        logger.info(f"No root passed. Chose {root} randomly.")
        T = Graph({root}, {root: self.node_labels.get(root)}, {}, {root: set()})
        T.root = root
        leaves = [root]
        while leaves:
            leaves = self._grow_tree(T, leaves) # until there are no more free leaves
        return T

    def get_spanning_tree_2(self, root=None, skip=None, 
            confidence=1, depth=0, rng=GRAPH_RNG) -> Graph: 
        """
        Find a spanning tree for the Graph by depth-first iteration, with a 
        random choice at each node whether to proceed or "back up."
        May terminate without finding a spanning set.
        """
        if skip is None:
            skip = set()
        nodes = set.difference(self.nodes, skip)
        #Check if {nodes} is empty
        if not nodes: 
            logger.debug("Empty!")
            return Graph.empty()
        
        # just pick lowest index for root
        root = min(nodes) if root is None else root
        skip.add(root) # persists when ascending the callstack

        # init punctual graph
        T = Graph(
            {root}, 
            {root: self.node_labels.get(root)}, 
            {},
            {root: set()}
        )
        T.root = root

        if depth and not rng.binomial(1, confidence):
            # back up to previous node, preserving skip set
            return T

        # join children
        for node in self.children[root]:
            if node in skip: continue
            
            T.join_with(
                self.get_spanning_tree_2(
                    root=node, 
                    skip=skip,
                    confidence=confidence,
                    depth=depth+1,
                    rng=rng
                ),
                at=(root, node),
                label=self.edge_labels[(root,node)]
            )
        return T

    def get_spanning_tree(self, root=None, skip=None) -> Graph: 
        """
        Find a spanning tree for the Graph.
        """
        if skip is None:
            skip = set()
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

    def iter_from(self, start: int):
        yield start
        for child in self.children[start]:
            for i in self.iter_from(child):
                yield i
