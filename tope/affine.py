from collections import namedtuple

# linear <- nxn matrix
# translate <- n-vector
AffTrans = namedtuple("AffTrans", ["linear", "translation"])

def apply(vecs, aff: AffTrans):
    """Apply affine transformation to a numpy stack of vectors."""
    return (vecs @ aff.linear) + aff.translation
