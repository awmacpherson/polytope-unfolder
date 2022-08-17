# Technical description

A Python package for unfolding polytope nets.

## The problem

Suppose we are given a polytope $\Delta$ in $n$ dimensions as a set $\mathrm{Vert}(\Delta)$ of vertices ($n$-vectors). By definition, a *net* for $\Delta$ is a tree $T$ together with a labelling of its nodes by:
- Facets of $\Delta$ (bijectively);
- A linear embedding of that facet into $\mathbb{R}^{n-1}$ (Euclidean $n-1$-space)

and a labelling of edges by $(n-2)$-faces such that

- If a pair of facets overlap, their intersection is exactly one face.
- If a pair of facets overlap in codimension one (i.e. in an $(n-2)$-face), then the corresponding nodes of $T$ are connected by an edge labelled by this face.


### What kind of projection?

We need to make some decisions about what kind of structure our projections and unfolding transformations are going to preserve. They should certainly be linear; they could also preserve the integral structure or be an orthogonal projection.

The important thing for the actual underlying mathematical objects is the integral structure. Strictly preserving this won't necessarily lead to nice models.

The coordinate system we use to define orthogonal projection, on the other hand, is fairly arbitrary from a mathematical perspective. However, it may be more visually satisfying and also be easier to construct nets whose facets are approximately "round" (as opposed to extremely elongated or pointy). This seems best ensured by paying attention to metric properties of the projection and reflections. 

## The algorithm

We can extract a "net" in $(n-1)$ dimensions as follows:

1. Compute the set of $k$-dimensional faces of $\Delta$ as a subset of $\mathrm{Pow}(\mathrm{Vert}(\Delta))$, for $k=0,\ldots,n$. Note that the $0$-faces are all the singleton sets, the $1$-faces all have exactly two elements, and the unique $n$-face is the whole of $\mathrm{Vert}(\Delta)$. For our algorithm, we actually only need knowledge of the $(n-1)$- and $(n-2)$-faces, but it is no more difficult to compute the faces of all dimensions.
2. Compute the incidence graph of the facets (i.e. $(n-1)$-faces).
3. Construct a rooted spanning tree of the incidence graph.
4. Iterate over the nodes of the tree (in any order, but depth-first seems likely to be the most optimised), "unfolding" as you go. This process is explained in more detail below.

### Computing the set of faces.

This process is probably already implemented in Magma, but I did this in Python so I had to write my own thing (which seems faster than learning Magma). The current algo is as follows:

1. Compute the set of facets. There are two approaches for this:
   - Compute the set of supporting hyperplanes for the polytope, that is, the dual representation. There are ready-made packages for this; I used a Python wrapper of the `qhull` algorithm. This loses the integral structure.
   - Find all maximal extremal sets of vertices. This seems like a potentially better approach but I haven't implemented it yet.
2. For $k=n-1,n-2,\ldots,1$, do the following:
   - Record all pairwise intersections of $k$-faces with facets, skipping degenerate cases where the $k$-face is contained in the facet.
   - Iterate over the resulting list and remove all items that are contained in another item.
   - Save what's left as the list of $(k-1)$-faces.

### Computing the incidence graph

This step is easy: iterate over all tuples of two facets and a codimension two face and see if they are incident. 

The incidence graph should be represented in a form optimised for constructing a spanning tree. The set of vertices is given by the previous step: it is the set of facets. Then we can do an exhaustive iteration to construct a mapping `vertices => set_of_neighbours`.

### Constructing a spanning tree

Another easy step. There's an algorithm for this.

### The **Unfolding**

Project our polytope into 3d. We now have a mapping from facets to polyhedra embedded in 3d, but they overlap, so we need to "unfold" them. We achieve this by picking a starting facet (labelled by the root of our spanning tree) and working our way out along the branches, checking at each node to see if the corresponding facet is "folded" and applying a reflection if so.

Let's make this a bit more precise. An edge of the incidence graph is labelled by two facets $F_1,F_2$ (which each label an end of that edge) of $\Delta$ and a common face $G=F_1\cap F_2$. If we are dealing with an edge from the spanning tree, then the root gives us an orientation of the edge (pointing away from the root); we make the convention that $F_1$ corresponds to the "parent" node and $F_2$ to the "child". Call these data an *incidence*.

The codimension two face $G$ spans an $(n-2)$-dimensional affine subspace $\langle G\rangle_\mathbb{R}$ in $\R^{n-1}$. This affine subspace is computed by solving an underdetermined matrix equation which gives us a defining function (covector) and an offset. Because it is a real hyperplane, it separates $\R^{n-1}$ into two components. The incidence $(F_1,F_2,G)$ is said to be *folded* if $F_1$ and $F_2$ are in the same component, i.e. they are on the "same side" of $\langle G\rangle_\R$.

To *unfold* a folded incidence, we reflect $F_2$ in the hyperplane $\langle G\rangle_\R$. To define a reflection, we need the hyperplane plus a normal vector. For this normal vector we could compute the orthogonal, but it seems easier and not obviously worse to just transpose the defining equation of the hyperplane.

So we are going to need library functions:
```
function affine_span(G: Face) -> (eq: float[3], offset: float)
function is_folded(F_1: Facet, F_2: Facet, G: Face) -> bool
function get_reflection(eq: float[3], offset: float) -> (lin: float[3,3], translation: float[3])
```

These reflections stack, so the actual algorithm is as follows:
1. For each node $p$ of $\Gamma$, walk from $\mathrm{root}(\Gamma)$ to $p$ and construct a sequence of affine transformations of length $d(\mathrm{root},p)$ where the transformation associated to an incidence $(F_1,F_2,G)$ is:
```
get_reflection(affine_span(G)) if is_folded(F_1, F_2, G) else Identity
```
2. Compose the embedding with the composite of these reflections (root-first order) to obtain our net.

This algorithm is most simply implemented with depth-first iteration, passing the sequence of reflections down the call stack.

## @Tom

As you say, the unfolding algorithm proceeds by extracting a spanning tree of the dual graph. This tree has vertices labelled by facets and edges labelled by codimension 2. I then do a reverse (i.e. starting from leaves) depth-first iteration over this tree (implemented as a recursion), where at each stage we do:
- compute the (affine) rotation transformation needed to get the child facet into the hyperplane of the parent (with suitable orientation); this matrix and translation is returned by rotate_into_hyperplane().
- Apply the rotation to the child facet and all its descendents.
The transformations are in R^N \rtimes SO(N) so they should still all fit together if folded up again (we can apply such transformations to the objects "by hand" after they are printed out and folded up!)
After doing all this, I compute an orthonormal basis of the subspace spanned by the net and reencode all the vertices in this basis. Orientation is fixed by saying that combining this basis with the inward pointing normal to the facet yields an oriented basis of R^N.

In the case of our 4d polytopes T, the full pipeline is to extract each 3d facet F in turn (as a standalone Tope object embedded in 3d), label the (2-)faces with the index of the corresponding 2-face of T, and apply the unfolding routine to F to get a 2d net with labelled cells. Orientations are fixed using the inward normal to F as above.
