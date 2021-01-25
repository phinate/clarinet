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

from dataclasses import dataclass
from dataclasses import field

import jax.numpy as jnp
from jax.interpreters.xla import _DeviceArray as JaxArray  # type: ignore


# funky inheritance pattern to keep defaults seperated from required args. see:
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
@dataclass(frozen=True)
class _Node:
    name: str


@dataclass(frozen=True)
class _NodeDefaults:
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    display_text: str = field(default="")


@dataclass(frozen=True)
class Node(_NodeDefaults, _Node):
    pass


# discrete
@dataclass(frozen=True)
class _Discrete(_Node):
    pass


@dataclass(frozen=True)
class _DiscreteDefaults(_NodeDefaults):
    prob_table: JaxArray = jnp.array([])


@dataclass(frozen=True)
class DiscreteNode(Node, _DiscreteDefaults, _Discrete):
    pass


# categorical
@dataclass(frozen=True)
class _Categorical(_Discrete):
    categories: List[Any]


@dataclass(frozen=True)
class _CategoricalDefaults(_DiscreteDefaults):
    pass


@dataclass(frozen=True)
class CategoricalNode(DiscreteNode, _CategoricalDefaults, _Categorical):
    pass
