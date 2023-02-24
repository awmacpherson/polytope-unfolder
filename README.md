# Tope.py

A Python package for unfolding polytope nets.

## Getting the polytope data

WARNING: This script currently does not work because the link to the polytope database is broken.

Run the following command after installing:
```sh
tope-get $NUM_POLYS > $FILENAME.json
```
where `$NUM_POLYS` is the desired number of polytopes and $FILENAME is the desired filename. 

## The problem

Suppose we are given a polytope $\Delta$ in $n$ dimensions as a set $\mathrm{Vert}(\Delta)$ of vertices ($n$-vectors). By definition, a *net* for $\Delta$ is a tree $T$ together with a labelling of its nodes by:
- Facets of $\Delta$ (bijectively);
- A linear embedding of that facet into $\mathbb{R}^{n-1}$ (Euclidean $n-1$-space)

and a labelling of edges by $(n-2)$-faces such that

- If a pair of facets overlap, their intersection is exactly one face.
- If a pair of facets overlap in codimension one (i.e. in an $(n-2)$-face), then the corresponding nodes of $T$ are connected by an edge labelled by this face.

## The algorithm

We can extract a "net" in $(n-1)$ dimensions as follows:

1. Compute the set of $k$-dimensional faces of $\Delta$ as a subset of $\mathrm{Pow}(\mathrm{Vert}(\Delta))$, for $k=0,\ldots,n$. Note that the $0$-faces are all the singleton sets, the $1$-faces all have exactly two elements, and the unique $n$-face is the whole of $\mathrm{Vert}(\Delta)$. For our algorithm, we actually only need knowledge of the $(n-1)$- and $(n-2)$-faces, but it is no more difficult to compute the faces of all dimensions.
2. Compute the incidence graph of the facets (i.e. $(n-1)$-faces).
3. Construct a rooted spanning tree of the incidence graph.
4. Iterate over the nodes of the tree (in any order, but depth-first seems likely to be the most optimised), "unfolding" as you go. This process is explained in more detail below.

### Computing the set of faces.

The first thing we need to do is compute the combinatorics of faces of the polytope from the set of vertices. This process is probably already implemented in Magma, but it seemed simpler to implement my own thing in Python than to learn Magma. 

The current algo is as follows:

1. Compute the set of facets. For this we use the [`pypoman`](https://pypi.org/project/pypoman) library to find a set of supporting hyperplanes, then intersect these with the vertex set to find the combinatorial facets. 
2. For $k=n-2,n-3,\ldots,1$, do the following:
   - Record all pairwise intersections of $k$-faces with facets, skipping degenerate cases where the $k$-face is contained in the facet.
   - Iterate over the resulting list and remove all items that are contained in another item.
   - Save what's left as the list of $(k-1)$-faces.

This algorithm is implemented in the `Tope.from_vertices(v)` method, which accepts a 2D `numpy` array whose rows are interpreted as the vertices and returns a Tope object, which records the vertices together with the combinatorics of faces as sets of indices into `v`.

### Computing the incidence graph

This step is easy: iterate over all pairs of facets and see if they intersect in a codimension two face. 

The incidence graph should be represented in a form optimised for constructing a spanning tree. The set of vertices is given by the previous step: it is the set of facets. Then we can do an exhaustive iteration to construct a mapping `vertices => set_of_neighbours`.

This step is implemented with the function `P.facet_graph()`, which returns a `Graph` object.

### Constructing a spanning tree

There are many ways to do this, and the method we choose will control the "layout" of the nets we produce. This package includes two algorithms:

- `Graph.get_spanning_tree()` implements a depth-first iteration. This produces "long" nets.
- `Graph.width_first_spanning_tree()`, as its name suggests, implements a width-first iteration. This produces "splayed out" nets.

### The **Unfolding**

We now have the data of a `Net` object, which is essentially that of the tree constructed in the previous step together with:
- nodes labelled by `Tope` objects constructed from the corresponding facets.
- edges labelled by the sets of indices that define the intersection $k-2$-cell.
We now need to "unfold" this net so that it fits into 3D. We achieve this by starting at the outermost leaves of our tree and working inwards, as we go applying rotations to get the descendants of a given facet into the same hyperplane as that face.

This procedure is implemented by the function `N.unfold()`, where `N` is a `Net` object.
