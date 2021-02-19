# clarinet 🎷*
*pending existence of clarinet emoji. for now, enjoy this rad sax.
![tests](https://github.com/phinate/clarinet/workflows/tests/badge.svg) [![codecov](https://codecov.io/gh/phinate/clarinet/branch/main/graph/badge.svg?token=ZBHFNPEP9R)](https://codecov.io/gh/phinate/clarinet) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/phinate/clarinet/main.svg)](https://results.pre-commit.ci/latest/github/phinate/clarinet/main)


A functional implementation of probabilistic networks.

** Note: This project is in pre-alpha, so expect the whole thing to be one sharp edge. **

## usage:

Here's a basic look at the DAG-making functionality. Keep in mind that all methods return a modified copy of the original object to avoid the headaches that come with mutated internal states :)

```py

import clarinet as cn


# bnlearn-style modelstring init
cn.BayesNet.from_modelstring("[A][C][B|A][D|C][F|A:B:C][E|F]")
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
        "categories": ["raining", "not raining"],
    },
    "cloudy": {
        "children": ["raining"],
    },
    "wet grass": {
        "parents": ["raining"],
    },
}

net = cn.BayesNet.from_dict(example_model_dict)
net
#> BayesNet(
#    nodes=<immutables.Map(
#        {
#            'wet grass': Node(name='wet grass', parents=('raining',), children=()),
#            'cloudy': Node(name='cloudy', parents=(), children=('raining',)),
#            'raining': CategoricalNode(name='raining', parents=('cloudy',), children=('wet grass',), prob_table=array([], dtype=float32), categories=('raining', 'not raining'))
#        }
#    ) at 0x16a6b2100>, modelstring=''
#)


# index into the network by name to look at a particular node
net["raining"]
#> CategoricalNode(name='raining', parents=('cloudy',), children=('wet grass',), prob_table=array([], dtype=float32), categories=('raining', 'not raining'))

# let's add some category names that we forgot!
net.convert_nodes(
    names=["wet grass", "cloudy"],
    new_node_types=[
        cn.CategoricalNode,
        cn.CategoricalNode
    ],
    new_node_kwargs=[
        dict(categories=["wet", "dry"]),
        dict(categories=["cloudy", "clear"]),
    ]
)
#> BayesNet(
#    nodes={
#        'wet grass': CategoricalNode(name='wet grass', parents=('raining',), children=(), prob_table=array([], dtype=float32), categories=('wet', 'dry')),
#        'cloudy': CategoricalNode(name='cloudy', parents=(), children=('raining',), prob_table=array([], dtype=float32), categories=('cloudy', 'clear')),
#        'raining': CategoricalNode(name='raining', parents=('cloudy',), children=('wet grass',), prob_table=array([], dtype=float32), categories=('raining', 'not raining'))
#    }, modelstring=''
#)
```
