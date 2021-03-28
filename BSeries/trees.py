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

def make_nodelist(root):
    """
    Recursively finds all the nodes that are descendants of root and returns
    them as a list.
    """
    nodelist = [root]
    if root.children is not None:
        for child in root.children:
            nodelist += make_nodelist(child)
    return nodelist

class RootedTree(object):
    r"""
    Represents a rooted tree.  Maintains two equivalent representations,
    one as a nested list and the other as a set of Nodes each with a
    parent and zero or more children.

    Can be initialized in three ways:

        1. From a nest list.
        2. From a set of Nodes.
        3. From a level sequence.

    To do: add examples.
    """

    def __init__(self,initializer):
        import numpy as np

        if initializer == []:
            self._nl = []
            self.root = Node(children=None,label=0)
            self.nodes = make_nodelist(self.root)

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

        elif type(initializer[0]) is Node:
            # Initialize from nodelist
            self.nodes = initializer
            self.root = self.nodes[0]
            # Now form nested list
            self._nl = []
            if self.root.children:
                for child in self.root.children:
                    self._nl.append(generate_nested_list(child,self.nodes))

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

            self.nodes = make_nodelist(self.root)

    def subtrees(self):
        return [RootedTree(nl) for nl in self._nl]

    def __len__(self):
        return len(self.nodes)

    def order(self):
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

        **Reference**: :cite:`butcher2003` p. 127, eq. 301(c)
        """
        gamma = len(self)
        for tree in self.subtrees():
            gamma *= tree.density()
        return gamma

    def symmetry(self):
        r"""
        The symmetry $\\sigma(t)$ of a rooted tree is...

        **Examples**::

            >>> from BSeries import trees
            >>> t = t.RootedTree([[],[],[[],[[]]]])
            >>> tree.symmetry()
            2

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

    def __eq__(self, tree2):
        return self._nl == tree2._nl

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
        """
        assert(n>=1)
        treecopy = RootedTree(self._nl)
        # Remove children that are not in primary tree
        for node in treecopy.nodes[:n]:
            for node2 in treecopy.nodes[n:]:
                if node.children:
                    if node2 in node.children:
                        node.children.remove(node2)
        # Add primary tree to forest
        forest = [RootedTree(treecopy.nodes[:n])]
        # Add secondary trees to forest
        for node in treecopy.nodes[n:]:
            if node.parent in treecopy.nodes[:n]:  # This node is an orphan
                # Form the tree with this orphan as root
                tree_nodes = make_nodelist(node)
                forest.append(RootedTree(tree_nodes))

        return forest


def plot_labeled_tree(nodelist):
    # Plot based on node structure
    import matplotlib.pyplot as plt
    plt.scatter([0],[0],c='k',marker='o')
    root = nodelist[0]
    if root.children is not None: _plot_labeled_subtree(nodelist,0,0,0,1.)
    xlim = plt.xlim()
    if xlim[1]-xlim[0]<1:
        plt.xlim(-0.5,0.5)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')


def _plot_labeled_subtree(nodelist, root_index, xroot, yroot, xwidth):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.scatter(xroot,yroot,c='k')
    plt.text(xroot+0.05,yroot,str(root_index),ha='left')
    root = nodelist[root_index]
    if root.children is None: return
    ychild = yroot + 1
    nchildren = len(root.children)
    dist = xwidth * (nchildren-1)/2.
    xchild = np.linspace(xroot-dist, xroot+dist, nchildren)
    for i, child in enumerate(root.children):
        plt.plot([xroot,xchild[i]], [yroot,ychild], '-k')
        ichild = nodelist.index(child)
        _plot_labeled_subtree(nodelist,ichild,xchild[i],ychild,xwidth/3.)

def generate_all_labelings(nodelist):
    """
    Returns a list of all permissible orderings of the nodes.  A permissible
    ordering is one in which each child comes after its parent.
    """
    labeled = [[nodelist[0]]]
    for i in range(1,len(nodelist)):
        new_labeled = []
        for base in labeled:
            for node in nodelist[1:]:
                if (node.parent in base) and (node not in base):
                    basecopy = base.copy()
                    basecopy.append(node)
                    new_labeled += [basecopy]
        labeled = new_labeled.copy()
    return labeled

def generate_distinct_labelings(nodelist):
    """
    Returns a list of all non-equivalent permissible orderings of the nodes.  A
    permissible ordering is one in which each child comes after its parent.
    Two orderings are equivalent if they differ only in the labels of 
    equivalent nodes.  Two nodes are equivalent if they have the same parent
    and identical subtrees.
    """
    labeled = [[nodelist[0]]]
    equivalence_set = []
    for i, node in enumerate(nodelist):
        es = []
        if i == 0: equivalence_set.append(es)
        else:
            nlform = generate_nested_list(node,nodelist)
            for child in node.parent.children:
                if generate_nested_list(child,nodelist)==nlform and nodelist.index(child) < i:
                    es.append(child)
            equivalence_set.append(es)
        
    for i in range(1,len(nodelist)):
        new_labeled = []
        for base in labeled:
            for node in nodelist[1:]:
                if (node.parent in base) and (node not in base):
                    if all([n in base for n in equivalence_set[nodelist.index(node)]]):
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
    """
    forest = []
    s = list(range(order))
    while s is not None:
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
