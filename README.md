This package is now deprecated; please us [bseries.jl](https://github.com/ranocha/bseries.jl) instead.  This Python implementation
was an initial prototype.  It contains relatively inefficient implementations and much
less functionality than the Julia version; it is also less tested and likely to contain
bugs.  I have left most of the old README intact below, but this package is not supported and
I most likely won't respond to requests.

# Bseries.py


`bseries.py` is a package for manipulating and computing with B-series in Python.
B-series are a tool for understanding and working with the structure of numerical
integrators; they have also found application in other areas.

At present,  the most useful tools in the package are implementations of
the composition and substitution laws for B-series.  In principle these
can be used to manipulate B-series up to any order, although performance
is currently an issue at high orders.  One of the most interesting applications
of this is the generation of modified equations for numerical discretizations
of nonlinear ODEs.  See these notebooks for some examples:


- [Modified equations analysis examples](https://nbviewer.jupyter.org/gist/ketch/28d83ec4134e62f8bd8ec4b3b6cccc4a)
- [Analysis of energy-conserving explicit discretization](https://nbviewer.jupyter.org/gist/ketch/03a99cdf8ef7d12860111f7a2dd58e23)

The package also implements multiples representations of rooted trees.

## Installation

```
pip install bseries
```
