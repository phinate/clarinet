# things to implement:

*note*: no core functionality exists for statistics yet! soon to be added :3

## nets:
- automatically generate modelstring from node map
- use a lib that displays PGMs, e.g.
    - [`daft`](https://docs.daft-pgm.org/en/latest/)
        - pros: nice style, lightweight
        - cons: no algorithm for node layouts (could implement later on)
    - [`graphviz`](https://github.com/xflr6/graphviz)
        - pros: seems to have all needed functionality
        - cons: requires additional binary, can't include as dependency

## nodes:
- attach distributions to nodes (possibly related to probprog choice)
- add Continuous nodes
- further subclass `DiscreteNode` for cases like Poisson, etc
    - maybe infer node type from distribution, and stop subclassing? just have Discrete (b/c prob table), then a `distribution` variable

## algorithms:
- variable elimination
- hook into probprog under the hood for approx inference
