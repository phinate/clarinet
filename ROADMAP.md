# things to implement:

## model import/export
- include ability to go to/from HBNet models
- add probability tables to schema
    - make sure to be careful to get the dimensions right! see jax named tensors: https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html

## algorithms:
- variable elimination for inference over discrete networks
- hook into probprog under the hood for approx inference (long-term ambition)
    - `numpyro` would be a nice choice for this -- need to figure out how to map the appropriate inference method to the corresponding pyro semantic

## nets:
- add validation routines as pydantic validators
    - need to refactor cycle checking using adjacency matrix
- use a lib that displays PGMs, e.g.
    - [`daft`](https://docs.daft-pgm.org/en/latest/)
        - pros: nice style, lightweight
        - cons: no algorithm for node layouts (could implement later on)
    - [`graphviz`](https://github.com/xflr6/graphviz)
        - pros: seems to have all needed functionality, integrated into `numpyro` utils
        - cons: requires additional binary, can't include as dependency

## nodes:
- Do proper dimensionality checks with probability tables, and check they sum to 1 on the correct axes
- attach distributions to nodes (possibly related to probprog choice)
- add Continuous nodes
- further subclass `BaseDiscrete` for cases like Poisson, etc
    - maybe infer node type from distribution, and stop subclassing? just have Discrete (b/c prob table), then a `distribution` variable
