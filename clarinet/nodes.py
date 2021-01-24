__all__ = [
    "_Node",
    "_NodeDefaults",
    "Node",
    "_Discrete",
    "_DiscreteDefaults",
    "DiscreteNode",
    "_Categorical",
    "_CategoricalDefaults",
    "CategoricalNode",
]

from typing import List
from typing import Any

import jax.numpy as jnp
from jax.interpreters.xla import _DeviceArray as JaxArray  # type: ignore
from flax import struct  # type: ignore


# funky inheritance pattern to keep defaults seperated from required args. see:
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
@struct.dataclass
class _Node:
    name: str = struct.field(pytree_node=False)


@struct.dataclass
class _NodeDefaults:
    parents: List[str] = struct.field(pytree_node=False, default_factory=list)
    children: List[str] = struct.field(pytree_node=False, default_factory=list)
    display_text: str = struct.field(pytree_node=False, default="")


@struct.dataclass
class Node(_NodeDefaults, _Node):
    pass


# discrete
@struct.dataclass
class _Discrete(_Node):
    pass


@struct.dataclass
class _DiscreteDefaults(_NodeDefaults):
    prob_table: JaxArray = jnp.array([])


@struct.dataclass
class DiscreteNode(Node, _DiscreteDefaults, _Discrete):
    pass


# categorical
@struct.dataclass
class _Categorical(_Discrete):
    categories: List[Any] = struct.field(pytree_node=False)


@struct.dataclass
class _CategoricalDefaults(_DiscreteDefaults):
    pass


@struct.dataclass
class CategoricalNode(DiscreteNode, _CategoricalDefaults, _Categorical):
    pass
