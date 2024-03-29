from __future__ import annotations

__all__ = [
    "Node",
    "DiscreteNode",
]

from typing import Any, Dict, List, Tuple

import xarray as xr
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
    states: Tuple[str, ...]
    prob_table: Any

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {xr.DataArray: lambda t: t.to_dict()}  # for self.json()

    @validator("states", pre=True)
    def state_val(cls, v: Any) -> Tuple[str, ...]:
        assert (type(v) == tuple) or (
            type(v) == list
        ), "Type of states needs to be a list or tuple"
        assert len(v) >= 2, "Need at least two states!"

        return tuple(v)
