from __future__ import annotations

__all__ = [
    "Node",
    "DiscreteNode",
    "CategoricalNode",
]

from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
from chex import Array
from pydantic import BaseModel, validator


class Node(BaseModel):
    name: str
    parents: Tuple[str, ...] = ()
    children: Tuple[str, ...] = ()

    class Config:
        allow_mutation = False

    @validator("parents", "children")
    def to_tuple(cls, v: Any) -> Tuple[str, ...]:
        return tuple(v)

    def add_parents(
        self,
        parents: List[str] | Tuple[str, ...],
    ) -> Node:
        parents = list(parents) + list(self.parents)
        dct = self.dict()
        dct["parents"] = parents
        return self.__class__(**dct)

    def add_children(
        self,
        children: List[str] | Tuple[str, ...],
    ) -> Node:
        children = list(children) + list(self.children)
        dct = self.dict()
        dct["children"] = children
        return self.__class__(**dct)

    @classmethod
    def from_node(cls, node: Node, **node_kwargs: Dict[str, Any]) -> Node:
        return cls(**node.dict(), **node_kwargs)


class DiscreteNode(Node):
    prob_table: Array = jnp.array([])

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {Array: lambda t: t.tolist()}  # for self.json()

    @validator("prob_table", pre=True)
    def to_array(cls, arr: Any) -> Array:
        return jnp.asarray(arr)


class CategoricalNode(DiscreteNode):
    categories: Tuple[str, ...]

    class Config:
        allow_mutation = False

    @validator("categories", pre=True)
    def category_val(cls, v: Any) -> Tuple[str, ...]:
        assert (type(v) == tuple) or (
            type(v) == list
        ), "Type of categories needs to be a list or tuple"
        assert len(v) >= 2, "Need at least two categories!"

        return tuple(v)
