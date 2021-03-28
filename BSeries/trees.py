class Node(object):

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
    """
    nl = []
    if root.children is None: return nl
    for child in root.children:
        nl.append(generate_nested_list(child,nodes))
    return nl

def make_nodelist(root):

    nodelist = [root]
    if root.children is not None:
        for child in root.children:
            nodelist += make_nodelist(child)
    return nodelist

class RootedTree(object):
    """
    Represents a rooted tree.  Maintains two equivalent representations,
    one as a nested list and the other as a set of Nodes each with a
    parent and zero or more children.
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

    def __len__(self):
        return len(self.nodes)

    def order(self):
        return len(self.nodes)

    def density(self):
        "$gamma(t)$"
        gamma = len(self)
        for tree in self.subtrees():
            gamma *= tree.density()
        return gamma

    def subtrees(self):
        return [RootedTree(nl) for nl in self._nl]

    def symmetry(self):
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
        pass # Need to write this function


    def plot(self):
        import matplotlib.pyplot as plt
        plt.scatter([0],[0],c='k',marker='o')
        if self._nl != []: plot_subtree(self._nl,0,0,1.)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

    def plot_labeled(self):
        plot_labeled_tree(self.nodes)

    def split(self,n):
        """
        Split a tree into the tree formed by the first n nodes
        and the trees that remain after those nodes are removed.
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

def generate_all_labelings(tree):
    labeled = [[tree.nodes[0]]]
    for i in range(1,len(tree.nodes)):
        new_labeled = []
        for base in labeled:
            for node in tree.nodes[1:]:
                if (node.parent in base) and (node not in base):
                    basecopy = base.copy()
                    basecopy.append(node)
                    new_labeled += [basecopy]
        labeled = new_labeled.copy()
    return labeled

def generate_distinct_labelings(tree):
    labeled = [[tree.nodes[0]]]
    equivalence_set = []
    for i, node in enumerate(tree.nodes):
        es = []
        if i == 0: equivalence_set.append(es)
        else:
            nlform = generate_nested_list(node,tree.nodes)
            for child in node.parent.children:
                if generate_nested_list(child,tree.nodes)==nlform and tree.nodes.index(child) < i:
                    es.append(child)
            equivalence_set.append(es)
        
    for i in range(1,len(tree.nodes)):
        new_labeled = []
        for base in labeled:
            for node in tree.nodes[1:]:
                if (node.parent in base) and (node not in base):
                    if all([n in base for n in equivalence_set[tree.nodes.index(node)]]):
                        basecopy = base.copy()
                        basecopy.append(node)
                        new_labeled += [basecopy]
        labeled = new_labeled.copy()
    return labeled



def plot_subtree(nestedlist, xroot, yroot, xwidth):
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


def elementary_weight(tree, A, b):
    import numpy as np
    root = tree.nodes[0]
    if root.children is None:
        return sum(b)
    elif len(root.children) == 1:
        return sum([b[j]*subweight_vector(root.children[0], tree.nodes, A)[j] for j in range(len(b))])
    else:
        return sum([b[j]*np.prod([subweight_vector(child, tree.nodes, A) for child in root.children],0)[j] for j in range(len(b))])


def subweight_vector(node, nodelist, A):
    import numpy as np
    if node.children is None:
        return np.sum(A,1)
    elif len(node.children) == 1:
        return sum([A[:,j]*subweight_vector(node.children[0], nodelist, A)[j] for j in range(A.shape[0])])
    else:
        return sum([A[:,j]*np.prod([subweight_vector(child, nodelist, A) for child in node.children],0)[j] for j in range(A.shape[0])])

def all_trees(order):
    """
    Generate all distinct (unlabelled) rooted trees of the prescribed order.
    Uses the algorithm of 
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
    Assume root is level 0.
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
