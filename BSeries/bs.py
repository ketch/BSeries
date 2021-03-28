"""
A library for working with B-series.
"""
def elementary_weight(tree, A, b):
    """
    Compute the elementary weight corresponding to a given tree and set of
    Runge-Kutta coefficients.
    """
    import numpy as np
    root = tree.nodes[0]
    if root.children is None:
        return sum(b)
    elif len(root.children) == 1:
        return sum([b[j]*subweight_vector(root.children[0], tree.nodes, A)[j] for j in range(len(b))])
    else:
        return sum([b[j]*np.prod([subweight_vector(child, tree.nodes, A) for child in root.children],0)[j] for j in range(len(b))])


def subweight_vector(node, nodelist, A):
    """
    Used recursively to compute elementary weights.  Called from elementary_weight().
    """
    import numpy as np
    if node.children is None:
        return np.sum(A,1)
    elif len(node.children) == 1:
        return sum([A[:,j]*subweight_vector(node.children[0], nodelist, A)[j] for j in range(A.shape[0])])
    else:
        return sum([A[:,j]*np.prod([subweight_vector(child, nodelist, A) for child in node.children],0)[j] for j in range(A.shape[0])])


