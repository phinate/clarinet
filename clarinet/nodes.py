from __future__ import annotations

__all__ = [
    "Node",
    "DiscreteNode",
    "CategoricalNode",
]

from pydantic import BaseModel, validator

import jax.numpy as jnp
from jax.numpy import ndarray as JaxArray  # type: ignore
from jax.interpreters.xla import DeviceArray  # type: ignore


class Node(BaseModel):
    name: str
    parents: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    display_text: str = ""

    class Config:
        allow_mutation = False

    @validator('parents', 'children')
    def to_tuple(cls, v: list[str]) -> tuple[str, ...]:
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


class DiscreteNode(Node):
    prob_table: JaxArray = jnp.array([])

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {DeviceArray: lambda t: t.tolist()}  # for self.json()


class CategoricalNode(DiscreteNode):
    categories: tuple[str]

    class Config:
        allow_mutation = False

    @validator('categories')
    def to_tuple(cls, v: list[str]) -> tuple[str, ...]:
        return tuple(v)

    @classmethod
    def from_node(
        cls,
        node: Node,
        categories: list[str] | tuple[str, ...],
        prob_table: JaxArray = jnp.array([])
    ) -> CategoricalNode:
        return cls(
            **node.dict(),
            categories=categories,
            prob_table=prob_table
        )
