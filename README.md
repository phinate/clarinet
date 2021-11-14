# clarinet 🎵

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phinate/clarinet/HEAD?filepath=tutorial.ipynb)

[![Actions Status][actions-badge]][actions-link]
[![codecov](https://codecov.io/gh/phinate/clarinet/branch/main/graph/badge.svg?token=ZBHFNPEP9R)](https://codecov.io/gh/phinate/clarinet) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/phinate/clarinet/main.svg)](https://results.pre-commit.ci/latest/github/phinate/clarinet/main)

[![Documentation Status][rtd-badge]][rtd-link]
[![Code style: black][black-badge]][black-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

(another) Open-source probabilistic graphical model library. Designed to be easy to use and interoperable with an internally developed package at Bain and Company, Roma.
** Note: This project is in pre-alpha, so expect the whole thing to be one sharp edge. **

## functionality roadmap:
Completed:
- basic DAG creation and manipulation via `Node` class
- `BayesNet` class that wraps DAG logic 
- stringent runtime type-checking via `pydantic`
- graph vizualisation capability
- on-the-fly construction of junction tree
- probability table naming and manipulation via `xarray`
- exporting of network structure to lightweight json format
- network interoperability with:
  - bnlearn-style modelstrings
  - HBNet json specification (Bain package)

To-do:
- inference via probabilistic programming backend (e.g. numpyro)

## usage:

Here's a basic look at the DAG-making functionality:

```py

from clarinet import BayesNet


# bnlearn-style modelstring init
BayesNet.from_modelstring("[A][C][B|A][D|C][F|A:B:C][E|F]")
#> BayesNet(
#    nodes=<immutables.Map(
#        {
#            'C': Node(name='C', parents=(), children=('D', 'F')),
#            'F': Node(name='F', parents=('A', 'B', 'C'), children=('E',)),
#            'B': Node(name='B', parents=('A',), children=('F',)),
#            'D': Node(name='D', parents=('C',), children=()),
#            'A': Node(name='A', parents=(), children=('B', 'F')),
#            'E': Node(name='E', parents=('F',), children=())
#        }
#    ) at 0x16b5475c0>, modelstring=''
#)


# dict-style init
example_model_dict = {
    "raining": {
        "parents": ["cloudy"],
        "children": ["wet grass"],
        "states": ["raining", "not raining"],
    },
    "cloudy": {
        "children": ["raining"],
    },
    "wet grass": {
        "parents": ["raining"],
    },
}

net = BayesNet.from_dict(example_model_dict)
net
#> BayesNet(
#    nodes=<immutables.Map(
#        {
#            'wet grass': Node(name='wet grass', parents=('raining',), children=()),
#            'cloudy': Node(name='cloudy', parents=(), children=('raining',)),
#            'raining': DiscreteNode(name='raining', parents=('cloudy',), children=('wet grass',), prob_table=array([], dtype=float32), states=('raining', 'not raining'))
#        }
#    ) at 0x16a6b2100>, modelstring=''
#)


# index into the network by name to look at a particular node
net["raining"]
#> DiscreteNode(name='raining', parents=('cloudy',), children=('wet grass',), prob_table=array([], dtype=float32), states=('raining', 'not raining'))

# let's add some state names that we forgot!
net.convert_nodes(
    names=["wet grass", "cloudy"],
    new_node_types=[
        cn.DiscreteNode,
        cn.DiscreteNode
    ],
    new_node_kwargs=[
        dict(states=["wet", "dry"]),
        dict(states=["cloudy", "clear"]),
    ]
)
#> BayesNet(
#    nodes={
#        'wet grass': DiscreteNode(name='wet grass', parents=('raining',), children=(), prob_table=array([], dtype=float32), states=('wet', 'dry')),
#        'cloudy': DiscreteNode(name='cloudy', parents=(), children=('raining',), prob_table=array([], dtype=float32), states=('cloudy', 'clear')),
#        'raining': DiscreteNode(name='raining', parents=('cloudy',), children=('wet grass',), prob_table=array([], dtype=float32), states=('raining', 'not raining'))
#    }, modelstring=''
#)
```



[actions-badge]:            https://github.com/phinate/clarinet/workflows/CI/badge.svg
[actions-link]:             https://github.com/phinate/clarinet/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/clarinet
[conda-link]:               https://github.com/conda-forge/clarinet-feedstock
[codecov-badge]:            https://app.codecov.io/gh/phinate/clarinet/branch/main/graph/badge.svg
[codecov-link]:             https://app.codecov.io/gh/phinate/clarinet
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/phinate/clarinet/discussions
[gitter-badge]:             https://badges.gitter.im/https://github.com/phinate/clarinet/community.svg
[gitter-link]:              https://gitter.im/https://github.com/phinate/clarinet/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[pypi-link]:                https://pypi.org/project/clarinet/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/clarinet
[pypi-version]:             https://badge.fury.io/py/clarinet.svg
[rtd-badge]:                https://readthedocs.org/projects/clarinet/badge/?version=latest
[rtd-link]:                 https://clarinet.readthedocs.io/en/latest/?badge=latest
[sk-badge]:                 https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
