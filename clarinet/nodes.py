__all__ = [
    "Node",
    "DiscreteNode",
    "CategoricalNode",
]

from typing import List
from typing import Tuple
from typing import Union

from pydantic import BaseModel

import jax.numpy as jnp
from jax.numpy import ndarray as JaxArray  # type: ignore
from jax.interpreters.xla import DeviceArray  # type: ignore


class Node(BaseModel):
    name: str
    parents: Tuple[str] = ()  # type: ignore
    children: Tuple[str] = ()  # type: ignore
    display_text: str = ""

    class Config:
        allow_mutation = False

    def add_parents(
        self,
        parents: Union[List[str], Tuple[str]],
    ) -> 'Node':
        return self.copy(
            update={'parents': tuple(parents)},
            deep=True
        )

    def add_children(
        self,
        children: Union[List[str], Tuple[str]],
    ) -> 'Node':
        return self.copy(
            update={'children': tuple(children)},
            deep=True
        )


class DiscreteNode(Node):
    prob_table: JaxArray = jnp.array([])
    # for pydantic

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {DeviceArray: lambda t: t.tolist()}  # for self.json()


class CategoricalNode(DiscreteNode):
    categories: Tuple[str]

    class Config:
        allow_mutation = False

    @classmethod
    def from_node(
        cls,
        node: Node,
        categories: Union[List[str], Tuple[str]],
        prob_table: JaxArray = jnp.array([])
    ) -> 'CategoricalNode':
        return cls(
            **node.dict(),
            categories=tuple(categories),  # type: ignore
            prob_table=prob_table
        )
