from __future__ import annotations

__all__ = [
    "Node",
    "DiscreteNode",
    "CategoricalNode",
]

from pydantic import BaseModel, validator

from typing import Any

import jax.numpy as jnp
from jax.interpreters.xla import _DeviceArray  # type: ignore


class Node(BaseModel):
    name: str
    parents: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    display_text: str = ""

    class Config:
        allow_mutation = False

    @validator('parents', 'children')
    def to_tuple(cls, v: Any) -> tuple[str, ...]:
        return tuple(v)

    def add_parents(
        self,
        parents: list[str] | tuple[str, ...],
    ) -> Node:
        return self.copy(
            update={'parents': parents},
            deep=True
        )

    def add_children(
        self,
        children: list[str] | tuple[str, ...],
    ) -> Node:
        return self.copy(
            update={'children': children},
            deep=True
        )

    @classmethod
    def from_node(
        cls,
        node: Node,
        **node_kwargs: dict[str, Any]
    ) -> Node:
        return cls(
            **node.dict(),
            **node_kwargs
        )


class DiscreteNode(Node):
    prob_table: _DeviceArray = jnp.array([])

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {_DeviceArray: lambda t: t.tolist()}  # for self.json()

    @validator('prob_table', pre=True)
    def to_array(cls, arr: Any) -> _DeviceArray:
        return jnp.asarray(arr)


class CategoricalNode(DiscreteNode):
    categories: tuple[str, ...]

    class Config:
        allow_mutation = False

    @validator('categories', pre=True)
    def to_tuple(cls, v: Any) -> tuple[str, ...]:
        return tuple(v)
