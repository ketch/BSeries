"""
Unit tests for BSeries.trees.
"""
from BSeries import trees

t1 = trees.RootedTree([])
t2 = trees.RootedTree([[]])
t31 = trees.RootedTree([[],[]])
t32 = trees.RootedTree([[[]]])
t41 = trees.RootedTree([[],[],[]])
t42 = trees.RootedTree([[],[[]]])
t43 = trees.RootedTree([[[],[]]])
t44 = trees.RootedTree([[[[]]]])
t51 = trees.RootedTree([[],[],[],[]])
t52 = trees.RootedTree([[],[],[[]]])
t53 = trees.RootedTree([[],[[],[]]])
t54 = trees.RootedTree([[],[[[]]]])
t55 = trees.RootedTree([[[]],[[]]])
t56 = trees.RootedTree([[[],[],[]]])
t57 = trees.RootedTree([[[],[[]]]])
t58 = trees.RootedTree([[[[],[]]]])
t59 = trees.RootedTree([[[[[]]]]])

forest = [t1, t2, t31, t32, t41, t42, t43, t44, t51, t52, t53, t54, t55, t56, t57, t58, t59]

def test_alpha():
    for tree in forest:
        alpha1 = len(tree.all_labelings())/tree.symmetry()
        alpha2 = len(tree.distinct_labelings())
        assert(alpha1 == alpha2)
