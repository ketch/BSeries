"""
A library for working with B-series.
"""

indices = 'ijklmnpqrstuvwxyz'
import sympy
h = sympy.Symbol('h')
y = sympy.Symbol('y')

import numpy as np

def object_einsum(string, *arrays):
    """
    Simplified object einsum, without much error checking.
    Stolen from https://gist.github.com/seberg/5236560, with some modifications.
    We can remove this and rely directly on np.einsum after the related
    PR is merged to numpy.
    """
    import copy
    try:
        return np.einsum(string, *arrays)
    except TypeError:
        pass
    
    s = string.split('->')
    in_op = s[0].split(',')
    out_op = None if len(s) == 1 else s[1].replace(' ', '')

    in_op = [axes.replace(' ', '') for axes in in_op]
    all_axes = set()
    repeated_axes = set()
    
    for axes in in_op:
        list(repeated_axes.update(ax) for ax in axes if ax in all_axes)
        all_axes.update(axes)

    if out_op is None:
        out_op = set(sorted(all_axes))
        list(out_op.discard(rep_ax) for rep_ax in repeated_axes)
    else:
        all_axes.update(out_op)
    
    perm_dict = {_[1]: _[0] for _ in enumerate(all_axes)}
    
    dims = len(perm_dict)
    op_axes = []
    for axes in (in_op + list((out_op,))):
        op = [-1] * dims
        for i, ax in enumerate(axes):
           op[perm_dict[ax]] = i
        op_axes.append(op)
    
    op_flags = [('readonly',)] * len(in_op) + [('readwrite', 'allocate')]
    dtypes = [np.object_] * (len(in_op) + 1) # cast all to object

    nditer = np.nditer(arrays + (None,), op_axes=op_axes, flags=['buffered', 'delay_bufalloc', 'reduce_ok', 'grow_inner', 'refs_ok'], op_dtypes=dtypes, op_flags=op_flags)

    nditer.operands[-1][...] = 0
    nditer.reset()
    
    for vals in nditer:
        out = vals[-1]
        prod = copy.deepcopy(vals[0])
        #prod = vals[0]
        for value in vals[1:-1]:
            prod *= value
        out += prod
    
    return nditer.operands[-1]

class BSeries(object):

    def __init__(self, coeffs=None, F=None):
        import sympy
        self.coeffs = coeffs
        if F:
            self.F = F
        else:
            self.F = sympy.Function('F')

    def eval(self, order=2):
        import sympy
        from BSeries import trees
        if self.coeffs:
            value = y
            for i in range(1,order+1):
                value += h**i/sympy.factorial(i)*sum([t.alpha()*self.coeffs(t)*self.F(t) for t in trees.all_trees(i)])
            return value

    def print(self, order=2):
        from BSeries import trees
        if self.F and (self.coeffs is None):
            # print elementary differentials with unknown coeffs
            pass
        elif self.coeffs and self.F is None:
            # print elementary weights and unknown differentials
            outstr = r'y\\'
            for p in range(1,order+1):
                outstr += r' + h^{}/{}! ('.format(p,p)
                for i, tree in enumerate(trees.all_trees(p)):
                    outstr += str(tree.alpha()*self.coeffs(tree))
                    outstr += ' F(t_{{ {}{} }})(y)'.format(p,i+1)
                    outstr += r'+'
                outstr = outstr[:-1]
                outstr += r')\\'
            return outstr

class TreeMap(dict):

    def __init__(self, symbol='a'):
        self.symbol = symbol

    def __missing__(self, key):
        from BSeries import trees
        from sympy import Symbol
        tree_name = next(name for name, tree in trees.canonical_forest.items() if tree == key)
        tree_index = tree_name[1:]
        self[key]=sympy.Symbol(self.symbol+tree_index)
        return self[key]

    def __call__(self, key):
        return self[key]


def RK_BSeries(method):
    cfun = lambda tree: tree.density() * elementary_weight(tree, method.A, method.b)
    return BSeries(coeffs=cfun)

def generic_elementary_differential_string(tree):
    for i, node in enumerate(tree.nodes):
        node.label = indices[i]
    factors = []
    for node in tree.nodes:
        factor = 'f^{}'.format(node.label)
        if node.children:
            factor += '_{'
            for child in node.children:
                factor += child.label
            factor += '}'
        factors.append(factor)

    return ' '.join(factors)

def elementary_differential(tree, f, u):
    from sympy import Derivative as D
    N = len(f); M = len(u);
    F1 = np.empty((N,M), dtype=object)
    for i in range(N):
        for j in range(M):
            F1[i,j] = D(f[i],u[j])

    root = tree.nodes[0]
    if root.children is None:
        return f
    elif len(root.children) == 1:
        return sum([F1[:,j]*sub_differential(node.children[0], tree.nodes, f)[j] for j in range(N)])
    else:
        return
    pass


def elementary_weight_einsum(tree, A, b):
    # This code will work with plain np.einsum() after
    # https://github.com/numpy/numpy/pull/18053 gets merged
    import numpy as np
    c = np.sum(A,1)
    inds = []
    tensors = []
    counter = 0
    if tree.nodes[0].children is None:
        return sum(b)
    for i, node in enumerate(tree.nodes):
        if i==0: 
            tensors.append(b)
            inds.append(indices[0])
            node.label = indices[0]
            counter += 1
        else:
            if node.children:
                tensors.append(A)
                node.label = indices[counter]
                inds.append(node.parent.label + node.label)
                counter += 1
            else:
                tensors.append(c)
                inds.append(node.parent.label)
    return object_einsum(','.join(inds), *tensors)
    #return np.einsum(','.join(inds), *tensors, optimize='optimal')


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


def subs(b, a, t, args='trees'):
    """
    Returns the coefficient of t in the B-series that is formed
    by substituting the B-series b into the B-series a.
    """
    from BSeries import trees
    from functools import reduce
    from operator import mul

    forests, skeletons = t.all_partitions()
    expr = 0
    for forest, skeleton in zip(forests, skeletons):
        if args == 'trees':
            expr += reduce(mul,[b(tree) for tree in forest])*a(skeleton)        
        elif args == 'integers':
            expr += reduce(mul,[b(trees.intmap(tree)) for tree in forest])*a(skeleton)        
    return expr
