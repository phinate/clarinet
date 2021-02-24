from __future__ import annotations

__all__ = [
    "Node",
    "DiscreteNode",
    "CategoricalNode",
]

from pydantic import BaseModel, validator
from typing import Any

import numpy as np


class Node(BaseModel):
    name: str
    parents: tuple[str, ...] = ()
    children: tuple[str, ...] = ()

    class Config:
        allow_mutation = False

    @validator("parents", "children")
    def to_tuple(cls, v: Any) -> tuple[str, ...]:
        return tuple(v)

    def add_parents(
        self,
        parents: list[str] | tuple[str, ...],
    ) -> Node:
        parents = list(parents) + list(self.parents)
        dct = self.dict()
        dct["parents"] = parents
        return self.__class__(**dct)

    def add_children(
        self,
        children: list[str] | tuple[str, ...],
    ) -> Node:
        children = list(children) + list(self.children)
        dct = self.dict()
        dct["children"] = children
        return self.__class__(**dct)

    @classmethod
    def from_node(cls, node: Node, **node_kwargs: dict[str, Any]) -> Node:
        return cls(**node.dict(), **node_kwargs)


class DiscreteNode(Node):
    prob_table: np.ndarray = np.array([])

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: lambda t: t.tolist()}  # for self.json()

    @validator("prob_table", pre=True)
    def to_array(cls, arr: Any) -> np.ndarray:
        return np.asarray(arr)


class CategoricalNode(DiscreteNode):
    categories: tuple[str, ...]

    class Config:
        allow_mutation = False

    @validator("categories", pre=True)
    def category_val(cls, v: Any) -> tuple[str, ...]:
        assert (type(v) == tuple) or (
            type(v) == list
        ), "Type of categories needs to be a list or tuple"
        assert len(v) >= 2, "Need at least two categories!"

        return tuple(v)
