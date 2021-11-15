"""
A library for working with B-series.

We use the coefficient normalization that appears in 
HLW (1.23), and in CHV2010 on the middle of p. 411.
"""

indices = 'jklmpqrstuvwxyz'
import sympy
h = sympy.Symbol('h')
y = sympy.Symbol('y')

import numpy as np

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
        if self.coeffs and not isinstance(F,sympy.FunctionClass):
            value = y
            for i in range(1,order+1):
                value += h**i*sum([self.coeffs(t)*self.F(t)/t.symmetry() for t in trees.all_trees(i)])
            return value

class TreeMap(dict):
    """
    A mapping from the rooted trees to the real numbers.

    Let 0 denote the empty tree.
    If a(0)=1, the resulting set of maps form a group under the
    composition law (see bs.compose()).  This is known as the
    Butcher group.

    If a(0)=0, the resulting set of maps form a monoid under the
    substitution law (see bs.subs()).
    """

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

def elementary_weight_string(tree):
    for i, node in enumerate(tree.nodes):
        node.label = indices[i]
    factors = []
    for node in tree.nodes:
        if node.children:
            for child in node.children:
                factor = 'a_{{ {}{} }}'.format(node.label,child.label)
                factors.append(factor)

    if factors == []: return '1'
    return ' '.join(factors)

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

def elementary_differential(tree, f=None, y=None, evaluate=False):
    """
    Compute the elementary differential corresponding to a given tree.

    If f is not provided, the return value is a LaTeX string corresponding
    to the elementary differential in summation form (as in Table 2.2
    of HNW1993).

    If f and y are provided, then the derivatives are computed and
    the summations evaluated.

    Perhaps this should be rewritten to take just f and y as arguments and return
    a function on trees.
    """
    from BSeries.util import object_einsum
    from BSeries import trees
    from sympy import Derivative as D
    from sympy import tensor
    from sympy import simplify
    dba = tensor.array.derive_by_array

    # This code will work with plain np.einsum() after
    # https://github.com/numpy/numpy/pull/18053 gets merged
    if f is None:
        # f is just an undefined symbolic function.  Return a string.
        outstr = '$'
        for i, node in enumerate(tree.nodes):
            node.label = indices[i]
        for i, node in enumerate(tree.nodes):
            outstr+='f^'+node.label
            if node.children:
                outstr += '_{'
                for child in node.children:
                    outstr += child.label
                outstr += '}'
        outstr += '$'
        return outstr

    # Empty tree and single node are special cases
    if tree.nodes[0].children is None:
        return f
    elif tree == trees.RootedTree([]):
        return f

    # Compute tensors of partial derivatives
    max_der = max([len(node.children) for node in tree.nodes if node.children]) # Highest-order derivative tensor we need
    F = []
    if evaluate:  # This has been tested and matches the results of the second approach below,
                  # as far as that one goes.
        X = f
        for j in range(1,max_der+1):
            X = dba(X,y)
            permind = [j]+list(range(j))
            Y = sympy.permutedims(X,permind)
            F.append(Y.copy())
    else:
        N = len(f); M = len(y);
        if max_der >= 1:
            F.append(np.empty((N,M), dtype=object))
            for i in range(N):
                for j in range(M):
                    F[0][i,j] = D(f[i],y[j],evaluate=evaluate)
        if max_der >= 2:
            F.append(np.empty((N,M,M), dtype=object))
            for i in range(N):
                for j in range(M):
                    for k in range(M):
                        F[1][i,j,k] = D(f[i],y[j],y[k],evaluate=evaluate)
        if max_der >= 3:
            F.append(np.empty((N,M,M,M), dtype=object))
            for i in range(N):
                for j in range(M):
                    for k in range(M):
                        for l in range(M):
                            F[2][i,j,k,l] = D(f[i],y[j],y[k],y[l],evaluate=evaluate)
        if max_der >= 4:
            F.append(np.empty([N]+[M]*4, dtype=object))
            for i in range(N):
                for j in range(M):
                    for k in range(M):
                        for l in range(M):
                            for m in range(M):
                                F[3][i,j,k,l,m] = D(f[i],y[j],y[k],y[l],y[m],evaluate=evaluate)
        if max_der >= 5:
            F.append(np.empty([N]+[M]*5, dtype=object))
            for i in range(N):
                for j in range(M):
                    for k in range(M):
                        for l in range(M):
                            for m in range(M):
                                for n in range(M):
                                    F[4][i,j,k,l,m,n] = D(f[i],y[j],y[k],y[l],y[m],y[n],evaluate=evaluate)
        if max_der >= 6:
            raise NotImplementedError

    # Set up and perform tensor contraction
    inds = []
    tensors = []
    counter = 0
    for i, node in enumerate(tree.nodes):
        if node.children:
            tensors.append(F[len(node.children)-1])
        else:
            tensors.append(f)
        node.label = indices[counter]
        if node.parent:
            node.parent.label += indices[counter]
        counter += 1
    inds = [node.label for node in tree.nodes]
    return object_einsum(','.join(inds), *tensors)


def elementary_weight_einsum(tree, A, b):
    # This code will work with plain np.einsum() after
    # https://github.com/numpy/numpy/pull/18053 gets merged
    from BSeries.util import object_einsum
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
    if tree._nl == None:
        return 0

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

def compose(b, a, t):
    """
    Returns the coefficient corresponding to tree t in the B-series that is
    formed by composing the B-series a with the B-series b.

    See CHV2010 Section 3.1.

    Perhaps this should be rewritten to take just (b, a) and return a function that takes a tree.
    Could also use operator overloading.

    Examples::

        >>> from BSeries import trees, bs
        >>> a = bs.TreeMap('a')
        >>> b = bs.TreeMap('b')
        >>> b[trees.RootedTree(None)]=1
        >>> t = trees.RootedTree([])
        >>> bs.compose(b,a,t)
        a0*b1 + a1
        >>> t = trees.RootedTree([[],[[]]])
        >>> bs.compose(b,a,t)
        a0*b42 + a1*b1*b2 + a2*b1**2 + a2*b2 + a31*b1 + a32*b1 + a42
    """
    from BSeries import trees
    from functools import reduce
    from operator import mul

    forests, subtrees = t.all_splittings()
    expr = 0
    for forest, subtree in zip(forests, subtrees):
        expr += reduce(mul,[b(tree) for tree in forest])*a(subtree)
    return expr

def subs(b, a, t):
    """
    Returns the coefficient corresponding to tree t in the B-series that is
    formed by substituting the B-series b into the B-series a.

    See CHV2010 Section 3.2.

    Perhaps this should be rewritten to take just (b, a) and return a function that takes a tree.
    Could also use operator overloading.

    Examples::

        >>> from BSeries import trees, bs
        >>> a = bs.TreeMap('a')
        >>> b = bs.TreeMap('b')
        >>> b[trees.RootedTree(None)]=0
        >>> t = trees.RootedTree([])
        >>> bs.subs(b,a,t)
        a1*b1
        >>> t = trees.RootedTree([[[]]])
        >>> bs.subs(b,a,t)
        a1*b32 + 2*a2*b1*b2 + a32*b1**3
    """
    from BSeries import trees
    from functools import reduce
    from operator import mul

    forests, skeletons = t.all_partitions()
    expr = 0
    for forest, skeleton in zip(forests, skeletons):
        expr += reduce(mul,[b(tree) for tree in forest])*a(skeleton)        
    return expr

def modified_equation(y, f, A, b, order=2, return_coefficients=False):
    """
    Return the modified equation up to a prescribed order in h, for the Runge-Kutta
    method (A,b) applied to the differential equation y'(t) = f(y).

    In other words, this computes a function $f_h(y)$ such that the approximate
    solution given by the RK method applied to y'(t) = f(y) is the exact solution
    of $y'(t) = f_h(y)$.  The function $f_h$ is expressed as a power series in
    $h$ and the returned function is the truncation of that power series (at the
    specified order).
    
    If `return_coefficients` is set to `True`, the function returns the coefficients
    of the B-series as `TreeMap`.

    See for example Section 3.2 of CHV2010.
    """
    from BSeries import trees
    cf = trees.canonical_forest

    # Construct the B-series of the RK method
    a = TreeMap('a')

    forest = trees.trees_to_order(order)
    for tree in forest.values():
        a[tree] = elementary_weight(tree,A,b)

    # Exact solution B-series:
    e = lambda t: sympy.Rational(1)/trees.gamma(t)
    # Modified equation B-series
    B = TreeMap('b')

    B[cf['t1']] = a[cf['t1']]
    # Recursively solve subs(B, e)(t) = a(t)
    # This works because subs(B, e, t) = B(t) + lower order terms
    for p in range(2,order+1):
        for t in trees.all_trees(p):
            B[t] = a[t] - subs(B, e, t) + B[t]

    if f is None:
        # Return a representation in terms of an undetermined RHS f.
        # Ideally here we would use trees that are also sympy Symbols.
        # But there are multiple inheritance issues that I haven't worked out.
        F = sympy.Function('F')
        series = 0
        for p in range(1,order+1):
            for t in trees.all_trees(p):
                sym = sympy.Symbol(t.__repr__())
                series += h**(p-1)/t.symmetry() * B[t]*F(sym)
        return series

    if return_coefficients:
        return B

    else:
        series = np.zeros_like(f,dtype=object)
        for p in range(1,order+1):
            for t in trees.all_trees(p):
                series += h**(p-1)/t.symmetry() * B[t]*elementary_differential(t,f,y,evaluate=True)

    return series

def modifying_integrator(y, f, A, b, order=2, return_coefficients=False):
    """
    Compute a "modifying integrator" ODE up to a prescribed order in h, for the
    Runge-Kutta method (A,b) applied to the differential equation y'(t) = f(y).

    In other words, this computes a function $f_h(y)$ such that the approximate
    solution given by the RK method applied to y'(t) = f_h(y) is the exact solution
    of $y'(t) = f(y)$.  The function $f_h$ is expressed as a power series in
    $h$ and the returned function is the truncation of that power series (at the
    specified order).
    
    If `return_coefficients` is set to `True`, the function returns the coefficients
    of the B-series as `TreeMap`.

    See for example Section 3.2 of CHV2010.
    """
    from BSeries import trees
    cf = trees.canonical_forest

    # Construct the B-series of the RK method
    a = TreeMap('a')

    forest = trees.trees_to_order(order)
    for tree in forest.values():
        a[tree] = elementary_weight(tree,A,b)

    # Exact solution B-series:
    e = lambda t: sympy.Rational(1)/trees.gamma(t)
    # Modified equation B-series
    B = TreeMap('b')

    B[cf['t1']] = e(cf['t1'])
    # Recursively solve subs(B, a)(t) = e(t)
    # This works because subs(B, a, t) = B(t) + lower order terms
    for p in range(2,order+1):
        for t in trees.all_trees(p):
            B[t] = e(t) - subs(B, a, t) + B[t]

    if return_coefficients:
        return B

    series = np.zeros_like(f,dtype=object)
    for p in range(1,order+1):
        for t in trees.all_trees(p):
            if B[t] != 0:  # This is just for efficiency -- skip the computation if the coeff is zero
                if f is not None:
                    series += h**(p-1)/t.symmetry() * B[t]*elementary_differential(t,f,y,evaluate=True)
                else:
                    print(B[t]/t.symmetry(), t)
                    #series += h**(p-1)/t.symmetry() * B[t]*t#elementary_differential(t,f,y)

    return series
