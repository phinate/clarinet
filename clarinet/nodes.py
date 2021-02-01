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
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Any

from dataclasses import dataclass
from dataclasses import field
from dataclasses import asdict

import jax.numpy as jnp
from jax.interpreters.xla import _DeviceArray as JaxArray  # type: ignore


# funky inheritance pattern to keep defaults seperated from required args. see:
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
@dataclass(frozen=True)
class _Node:
    name: str


@dataclass(frozen=True)
class _NodeDefaults:
    # see https://github.com/python/mypy/issues/5738 for ignore reasons
    parents: Tuple[str] = field(default_factory=tuple)  # type: ignore # noqa
    children: Tuple[str] = field(default_factory=tuple)  # type: ignore # noqa
    display_text: str = field(default="")


@dataclass(frozen=True)
class Node(_NodeDefaults, _Node):
    def set_parents(
        self,
        parents: Union[List[str], Tuple[str]],
        keep_current: bool = True
    ) -> _Node:
        if keep_current:
            current = list(self.parents)
            new_parents = tuple(list(parents) + current)
        else:
            new_parents = tuple(parents)

        self_dict = asdict(self)
        self_dict['parents'] = new_parents
        return self.__class__(**self_dict)

    def set_children(
        self,
        children: Union[List[str], Tuple[str]],
        keep_current: bool = True
    ) -> _Node:
        if keep_current:
            current = list(self.children)
            new_children = tuple(list(children) + current)
        else:
            new_children = tuple(children)

        self_dict = asdict(self)
        self_dict['children'] = new_children
        return self.__class__(**self_dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# discrete
@dataclass(frozen=True)
class _Discrete(_Node):
    pass


@dataclass(frozen=True)
class _DiscreteDefaults(_NodeDefaults):
    prob_table: Union[JaxArray, List[float]] = jnp.array([])


@dataclass(frozen=True)
class DiscreteNode(Node, _DiscreteDefaults, _Discrete):
    def make_serializable(self) -> _Discrete:
        if isinstance(self.prob_table, list):
            print(f"Node '{self.name}' is already serializable!")
            return self
        self_dict = asdict(self)
        self_dict['prob_table'] = self_dict['prob_table'].tolist()
        return self.__class__(**self_dict)


# categorical
@dataclass(frozen=True)
class _Categorical(_Discrete):
    categories: Tuple[str]


@dataclass(frozen=True)
class _CategoricalDefaults(_DiscreteDefaults):
    pass


@dataclass(frozen=True)
class CategoricalNode(DiscreteNode, _CategoricalDefaults, _Categorical):
    @classmethod
    def from_node(
        cls,
        node: Node,
        categories: Union[List[str], Tuple[str]],
        prob_table: JaxArray = jnp.array([])
    ) -> _Categorical:
        node_dict = node.as_dict()
        return cls(
            **node_dict,
            categories=tuple(categories),  # type: ignore
            prob_table=jnp.array(prob_table)  # just in case...
        )
