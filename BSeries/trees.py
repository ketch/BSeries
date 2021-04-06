"""
Tools for representing and manipulating rooted trees.
The main component of this module is the RootedTree object,
whose documentation contains further details.

To facilitate the display of trees, and since it is not
convenient to use graph representations in most contexts,
the module also defines canonical names for (unlabeled)
trees up to order 5.  These names agree with those used
in the books of Hairer et. al. (e.g. HNW1993) and take
the form $t_{nj}$ where $n$ is the order of the tree and
$j$ is an index based on a chosen canonical ordering of the trees.
The canonical trees can be accessed directly via
the dictionary trees.canonical_forest.
"""
class Node(object):
    """
    A single node of a tree.

    Perhaps the label property is unneeded.
    """

    def __init__(self, parent=None, children=None, label=None):
        self.parent = parent
        self.children = children
        self.label = label

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)


def generate_child_nodes(ls, parent, counter):
    """
    Used recursively to generate the nodal representation of a tree.
    """
    for child in ls:
        counter += 1
        cn = Node(parent=parent, label=counter)
        parent.add_child(cn)
        counter = generate_child_nodes(child,cn, counter)

    return counter

def generate_nested_list(root,nodes):
    """
    Generates a nested list representation of the tree
    with specified node as root.  Useful for checking
    equality of trees or subtrees.

    To do: make the nested list form a property that is computed when needed.
    """
    nl = []
    if root.children is None: return nl
    for child in root.children:
        nl.append(generate_nested_list(child,nodes))
    return nl

def descendants(node):
    """
    Recursively finds all the nodes that are descendants of node and returns
    them as a list.
    """
    node_list = [node]
    if node.children is not None:
        for child in node.children:
            node_list += descendants(child)
    return node_list

class RootedTree(object):
    r"""
    Represents a rooted tree.  Maintains two equivalent representations,
    one as a nested list and the other as a set of Nodes each with a
    parent and zero or more children.  The tree is essentially viewed
    as unlabeled/unordered; two trees that are the same up to node
    labels are considered equal.
    But the order of the list of nodes (tree.nodes) can be used as a labeling.

    Can be initialized in three ways:

        1. From a nested list.
        2. From a set of Nodes.
        3. From a level sequence.

    Examples::

        >>> from BSeries import trees
        >>> t = trees.RootedTree([[],[],[[]]])
        >>> t
        t52
        >>> subtree = trees.RootedTree(trees.descendants(t.nodes[1]))
        >>> print(subtree)
        t1
    """
    def __init__(self, initializer, name=None):
        import numpy as np

        self.name = name

        if initializer == []:
            self._nl = []
            self.root = Node(children=None,label=0)
            self.nodes = descendants(self.root)

        elif initializer[0] == 0:
            # Initialize from level sequence
            assert(min(initializer)>=0)
            assert(min(np.diff(initializer))<=1)
            self.root = Node(children=None,label=0)
            self.nodes = [self.root]
            lev_seq = initializer
            for i, level in enumerate(lev_seq[1:]):
                ind = i+1
                iparent = max(loc for loc, val in enumerate(lev_seq[:ind]) if val == level-1)
                self.nodes.append(Node(parent=self.nodes[iparent]))
                self.nodes[iparent].add_child(self.nodes[-1])

            # Now form nested list
            self._nl = []
            if self.root.children:
                for child in self.root.children:
                    self._nl.append(generate_nested_list(child,self.nodes))
            self._nl = sorted_tree(self._nl)

        elif type(initializer[0]) is Node:
            # Initialize from node_list
            self.nodes = initializer
            self.root = self.nodes[0]
            # Now form nested list
            self._nl = []
            if self.root.children:
                for child in self.root.children:
                    self._nl.append(generate_nested_list(child,self.nodes))
            self._nl = sorted_tree(self._nl)

        else:   # Initialize from nested list
            self._nl = initializer  # Nested list form

            # Generate nodal form
            counter = 0
            self.root = Node(label=counter)
            for child in self._nl:
                counter += 1
                child_node = Node(parent=self.root, label=counter)
                self.root.add_child(child_node)
                counter = generate_child_nodes(child, child_node, counter)

            self.nodes = descendants(self.root)
            self._nl = sorted_tree(self._nl)

    def __hash__(self):
        # Required so that Trees can be dict keys
        return hash(to_tuple(sorted_tree(self._nl)))

    def __repr__(self):
        if self.name: return self.name
        else:
            nl = sorted_tree(self._nl)
            try:
                return next(name for name, tree in canonical_forest.items() if tree._nl == nl)
            except:
                return str(self._nl)

    def copy(self):
        return RootedTree(self._nl)

    def subtrees(self):
        r"""
        Returns a list of the trees that are left when the root of this tree is removed.

        Examples::

            >>> from BSeries import trees
            >>> t = trees.RootedTree([[],[[]]])
            >>> t.subtrees()
            [t1, t2]
        """
        return [RootedTree(nl) for nl in self._nl]

    def __len__(self):
        return len(self.nodes)

    def order(self):
        r"""
        The order of a tree is the number of nodes.
        """
        return len(self.nodes)

    def density(self):
        r"""
        The density of a rooted tree, denoted by $\\gamma(t)$,
        is the product of the orders of the subtrees.

        Examples::

            >>> from BSeries import trees
            >>> t = trees.RootedTree([[],[],[[],[[]]]])
            >>> t.density()
            56
            >>> t = trees.RootedTree([[],[],[]])
            >>> t.density()
            4


        **Reference**: :cite:`butcher2003` p. 127, eq. 301(c)
        """
        gamma = len(self)
        for tree in self.subtrees():
            gamma *= tree.density()
        return gamma

    def symmetry(self):
        r"""
        The symmetry $\\sigma(t)$ of a rooted tree is...

        Examples::

            >>> from BSeries import trees
            >>> t = RootedTree([[],[],[[],[[]]]])
            >>> t.symmetry()
            2
            >>> t = trees.RootedTree([[],[],[]])
            >>> t.symmetry()
            6

        **Reference**: :cite:`butcher2003` p. 127, eq. 301(b)
        """
        from sympy import factorial
        sigma = 1
        unique_subtrees = []
        for subtree in self._nl:
            if subtree not in unique_subtrees:
                unique_subtrees.append(subtree)
        for tree in unique_subtrees:
            m = self._nl.count(tree)
            sigma *= factorial(m)*RootedTree(tree).symmetry()**m
        return sigma

    def alpha(self):
        r"""
        The number of elements of the equivalence class of the tree;
        i.e., the number of possible different monotonic labelings.

        The formula used here comes from HLW p. 58, eqn. III.1.27.

        Examples::

            >>> from BSeries import trees
            >>> t = RootedTree([[],[],[[],[[]]]])
            >>> t.alpha()
            45
            >>> t = trees.RootedTree([[],[],[]])
            >>> t.alpha()
            1

        """
        from sympy import factorial
        return factorial(len(self))/(self.symmetry()*self.density())
        # This would also work:
        #return len(self.distinct_labelings())

    def __eq__(self, tree2):
        r"""
        Test equality of two rooted trees.

        Example::

            >>> from BSeries import trees as t
            >>> t.RootedTree([[],[[]]]) == t.RootedTree([[[]],[]])
            True
            >>> t.RootedTree([[],[]]) == t.RootedTree([[[]]])
            False
        """
        return sorted_tree(self._nl) == sorted_tree(tree2._nl)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.scatter([0],[0],c='k',marker='o')
        if self._nl != []: plot_subtree(self._nl,0,0,1.)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

    def plot_labeled(self):
        plot_labeled_tree(self.nodes)

    def all_labelings(self):
        return generate_all_labelings(self.nodes)

    def distinct_labelings(self):
        return generate_distinct_labelings(self.nodes)

    def split(self,n):
        """
        Split a labeled (ordered) tree into the tree formed by the first n
        nodes and the trees that remain after those nodes are removed.

        This operation is defined in HNW1993 p. 266, Definition 12.2.
        It can be used to perform the composition of two B-series.

        This is also described in Section 2.2 of CHV2010.
        """
        assert(n>=1)
        treecopy = RootedTree(self._nl)
        # Remove children that are not in primary tree
        for node in treecopy.nodes[:n]:
            for node2 in treecopy.nodes[n:]:
                if node.children:
                    if node2 in node.children:
                        node.children.remove(node2)
        subtree = RootedTree(treecopy.nodes[:n])

        # Secondary trees form forest
        forest = []
        for node in treecopy.nodes[n:]:
            if node.parent in treecopy.nodes[:n]:  # This node is an orphan
                # Form the tree with this orphan as root
                tree_nodes = descendants(node)
                forest.append(RootedTree(tree_nodes))

        return subtree, forest


    def all_partitions(self):
        """
        Compute all partitions of a tree, as defined in Section 2.3
        of CHV2010.
        """
        forests = []
        skeletons = []
        num_edges = len(self)-1
        for i in range(2**num_edges):  # Loop over all partitions
            edge_set = bin(i)[2:].zfill(num_edges)
            # edge_set is a string where '0' indicates that the edge leading to
            # the node in that place is removed.
            forest = partition_forest(self, edge_set)
            skeleton = partition_skeleton(self, edge_set)
            forests.append(forest)
            skeletons.append(skeleton)

        return forests, skeletons


def partition_forest(tree, edge_set):
    """
    The partition forest of the tree represented by node_list
    is the set of trees that result when the edges marked by
    zero in edge_set are removed.
    """
    import numpy as np
    # first make a copy
    node_list = tree.copy().nodes
    # Now sever the indicated edges
    for i, node in enumerate(node_list[1:]):
        if edge_set[i] == '0':
            node.parent.children.remove(node)
    # Now assemble the forest
    forest = []
    included = np.zeros(len(tree))
    for i, node in enumerate(node_list):
        if included[i] == 0:
            tree_nodes = descendants(node)
            forest.append(RootedTree(tree_nodes))
            for tree_node in tree_nodes:
                included[node_list.index(tree_node)] = 1
    return forest

def partition_skeleton(tree, edge_set):
    """
    The partition skeleton is the tree obtained by contracting each
    tree of the partition forest to a single vertex and then replacing
    the edges of the tree.

    The way this is performed here is: for each edge that is "kept"
    in edge_set, the child node is removed and its children are
    given to the parent (i.e. to their grandparent), recursively.
    """
    from copy import copy
    # first make a copy
    nodes = tree.copy().nodes
    nodes = nodes[::-1]  # We want to loop through children before parents
    edge_set = edge_set[::-1]
    for i, node in enumerate(nodes[:-1]):
        if edge_set[i] == '1':
            if node.children:
                node.parent.children += node.children
            node.parent.children.remove(node)

    nodes = nodes[::-1]
    edge_set = edge_set[::-1]
    skeleton_node_indices = [0] + [i+1 for i in range(len(edge_set)) if edge_set[i]=='0']
    skeleton_nodes = [nodes[i] for i in skeleton_node_indices]
    return RootedTree(skeleton_nodes)


def plot_labeled_tree(node_list):
    # Plot based on node structure
    import matplotlib.pyplot as plt
    plt.scatter([0],[0],c='k',marker='o')
    root = node_list[0]
    if root.children is not None: _plot_labeled_subtree(node_list,0,0,0,1.)
    xlim = plt.xlim()
    if xlim[1]-xlim[0]<1:
        plt.xlim(-0.5,0.5)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')


def _plot_labeled_subtree(node_list, root_index, xroot, yroot, xwidth):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.scatter(xroot,yroot,c='k')
    plt.text(xroot+0.05,yroot,str(root_index),ha='left')
    root = node_list[root_index]
    if root.children is None: return
    ychild = yroot + 1
    nchildren = len(root.children)
    dist = xwidth * (nchildren-1)/2.
    xchild = np.linspace(xroot-dist, xroot+dist, nchildren)
    for i, child in enumerate(root.children):
        plt.plot([xroot,xchild[i]], [yroot,ychild], '-k')
        ichild = node_list.index(child)
        _plot_labeled_subtree(node_list,ichild,xchild[i],ychild,xwidth/3.)

def generate_all_labelings(node_list):
    """
    Returns a list of all permissible orderings of the nodes.  A permissible
    ordering is one in which each child comes after its parent.

    See e.g. HNW1993 p. 267, Example 12.5.
    """
    labeled = [[node_list[0]]]
    for i in range(1,len(node_list)):
        new_labeled = []
        for base in labeled:
            for node in node_list[1:]:
                if (node.parent in base) and (node not in base):
                    basecopy = base.copy()
                    basecopy.append(node)
                    new_labeled += [basecopy]
        labeled = new_labeled.copy()
    return labeled

def generate_distinct_labelings(node_list):
    """
    Returns a list of all non-equivalent permissible orderings of the nodes.  A
    permissible ordering is one in which each child comes after its parent.
    Two orderings are equivalent if they differ only in the labels of 
    equivalent nodes.  Two nodes are equivalent if they have the same parent
    and identical subtrees.
    """
    labeled = [[node_list[0]]]
    equivalence_set = []
    for i, node in enumerate(node_list):
        es = []
        if i == 0: equivalence_set.append(es)
        else:
            nlform = generate_nested_list(node,node_list)
            for child in node.parent.children:
                if generate_nested_list(child,node_list)==nlform and node_list.index(child) < i:
                    es.append(child)
            equivalence_set.append(es)
        
    for i in range(1,len(node_list)):
        new_labeled = []
        for base in labeled:
            for node in node_list[1:]:
                if (node.parent in base) and (node not in base):
                    if all([n in base for n in equivalence_set[node_list.index(node)]]):
                        basecopy = base.copy()
                        basecopy.append(node)
                        new_labeled += [basecopy]
        labeled = new_labeled.copy()
    return labeled


def plot_subtree(nestedlist, xroot, yroot, xwidth):
    """
    Used recursively to plot unlabeled trees.
    Called by tree.plot().
    """
    import matplotlib.pyplot as plt
    import numpy as np
    plt.scatter(xroot,yroot,c='k')
    if nestedlist != []:
        ychild = yroot + 1
        nchildren = len(nestedlist)
        dist = xwidth * (nchildren-1)/2.
        xchild = np.linspace(xroot-dist, xroot+dist, nchildren)
        for i, child in enumerate(nestedlist):
            plt.plot([xroot,xchild[i]], [yroot,ychild], '-k')
            plot_subtree(child,xchild[i],ychild,xwidth/3.)


def plot_forest(forest):
    """
    Plot a collection of trees.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    nplots = len(forest)
    nrows=int(np.ceil(np.sqrt(float(nplots))))
    ncols=int(np.floor(np.sqrt(float(nplots))))
    if nrows*ncols<nplots: ncols=ncols+1
    #fig, axes = plt.subplots(nrows,ncols)
    for i, tree in enumerate(forest):
        plt.subplot(nrows,ncols,i+1)
        tree.plot_labeled()


def all_trees(order):
    """
    Generate all distinct (unlabelled) rooted trees of the prescribed order.
    Uses level sequences and the algorithm of 

      Beyer, Terry, and Sandra Mitchell Hedetniemi.
      "Constant time generation of rooted trees."
      SIAM Journal on Computing 9.4 (1980): 706-712.

    Note that the ordering of the trees matches the tables from
    Hairer's books up to order 4, but differs at order 5.
    """
    if order == 1: return [RootedTree([])]
    forest = []
    s = list(range(order))
    i = 0
    while s is not None:
        i += 1
        forest.append(RootedTree(s))
        s = get_successor(s)

    return forest[::-1]

def get_successor(s):
    """
    Compute the regular lexicographic successor to level sequence s.
    Assumes root is level 0.
    Called by all_trees().
    """
    import numpy as np
    if np.all(np.array(s[1:])-1==s[0]):
        return None
    else:
        p = max(loc for loc, val in enumerate(s) if val > 1)
        q = max(loc for loc, val in enumerate(s[:p]) if val == s[p]-1)  # Parent of s[p]
        successor = s[:p]
        Tq = s[q:p]
        while len(successor) <= len(s)-len(Tq):
            successor += Tq
        successor += Tq[:(len(s)-len(successor))]
        return successor

def sorted_tree(ls):
    """
    Recursively sort a nested list to get a canonical version.
    """
    for i in range(len(ls)):
        if type(ls[i]) is list:
            ls[i] = sorted_tree(ls[i])
    ls.sort()
    return ls

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def gamma(t):
    return t.density()

canonical_forest = {}
canonical_forest['t1'] = RootedTree([])
canonical_forest['t2'] = RootedTree([[]])
canonical_forest['t31'] = RootedTree([[],[]])
canonical_forest['t32'] = RootedTree([[[]]])
canonical_forest['t41'] = RootedTree([[],[],[]])
canonical_forest['t42'] = RootedTree([[],[[]]])
canonical_forest['t43'] = RootedTree([[[],[]]])
canonical_forest['t44'] = RootedTree([[[[]]]])
canonical_forest['t51'] = RootedTree([[],[],[],[]])
canonical_forest['t52'] = RootedTree([[],[],[[]]])
canonical_forest['t53'] = RootedTree([[],[[],[]]])
canonical_forest['t54'] = RootedTree([[],[[[]]]])
canonical_forest['t55'] = RootedTree([[[]],[[]]])
canonical_forest['t56'] = RootedTree([[[],[],[]]])
canonical_forest['t57'] = RootedTree([[[],[[]]]])
canonical_forest['t58'] = RootedTree([[[[],[]]]])
canonical_forest['t59'] = RootedTree([[[[[]]]]])
for p in range(6,10):
    for i, tree in enumerate(all_trees(p)):
        name = 't'+str(p)+str(i)
        canonical_forest[name] = tree
