# things to implement:

*note*: no core functionality exists for statistics yet! soon to be added :3

## algorithms:
- variable elimination for inference over discrete networks
- hook into probprog under the hood for approx inference (long-term ambition)

## nets:
- add validation routines as pydantic validators
    - to skip validation, use model.copy!
- ~automatically generate modelstring from node map~
- use a lib that displays PGMs, e.g.
    - [`daft`](https://docs.daft-pgm.org/en/latest/)
        - pros: nice style, lightweight
        - cons: no algorithm for node layouts (could implement later on)
    - [`graphviz`](https://github.com/xflr6/graphviz)
        - pros: seems to have all needed functionality
        - cons: requires additional binary, can't include as dependency

## nodes:
- Do proper dimensionality checks with probability tables, and check they sum to 1 on the correct axes
- attach distributions to nodes (possibly related to probprog choice)
- add Continuous nodes
- further subclass `DiscreteNode` for cases like Poisson, etc
    - maybe infer node type from distribution, and stop subclassing? just have Discrete (b/c prob table), then a `distribution` variable
