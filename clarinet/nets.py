from __future__ import annotations

__all__ = ["BayesNet"]

from typing import Any

from functools import singledispatchmethod
from pydantic import BaseModel, validator, validate_model, root_validator

from immutables import Map

import numpy as np

from .nodes import Node, CategoricalNode, DiscreteNode
from .validation import validate_node, validate_model_dict
from .utils import nodes_to_dict, modelstring_to_dict


class BayesNet(BaseModel):
    nodes: Map[str, Node]
    modelstring: str = ""

    # for pydantic
    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {
            Map: lambda t: {name: node for name, node in t.items()},
            np.ndarray: lambda t: t.tolist(),
        }
        keep_untouched = (singledispatchmethod,)

    def __getitem__(self, item: str) -> Node:
        return self.nodes[item]

    @validator("nodes", pre=True)
    def dict_to_map(cls, dct: dict[str, Node]) -> Map[str, Node]:
        return Map(dct)

    @root_validator
    def generate_modelstring(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values["modelstring"] == "":
            for node in values["nodes"].values():
                parents = node.parents
                values["modelstring"] += f"[{node.name}"
                values["modelstring"] += f"|{':'.join(parents)}]" if parents else "]"
        return values

    @classmethod
    def from_dict(
        cls,
        network_dict: dict[str, dict[str, Any]],
        validation: bool = True,
        modelstring: str = "",
    ) -> BayesNet:
        # validation step
        # TODO: jsonschema
        if validation:
            validate_model_dict(network_dict)
        nodes: dict[str, Node] = {}
        for i, items in enumerate(network_dict.items()):
            name, node_dict = items
            if "name" not in node_dict.keys():
                node_dict["name"] = name
            # special casing
            if "categories" in node_dict.keys():
                nodes[name] = CategoricalNode(**node_dict)
            elif "prob_table" in node_dict.keys():
                nodes[name] = DiscreteNode(**node_dict)
            else:
                nodes[name] = Node(**node_dict)
            # display_text = node.display_text or name  # TODO daft
        return cls(nodes=nodes, modelstring=modelstring)

    @classmethod
    def from_modelstring(cls, modelstring: str) -> BayesNet:
        return cls.from_dict(modelstring_to_dict(modelstring), modelstring=modelstring)

    @singledispatchmethod
    def add_node(self, node):  # type: ignore
        raise NotImplementedError(f"Type '{type(node)}' of node not recognised")

    # functools can't parse the return annotation here? ignore type for now
    @add_node.register
    def _(self, node: Node):  # type: ignore
        node_dct = dict(self.nodes)
        name = node.name
        node_dct[name] = node
        # make sure to complete arcs from their other ends
        for child in node.children:
            node_dct[child] = node_dct[child].add_parents([name])
        for parent in node.parents:
            node_dct[parent] = node_dct[parent].add_children([name])
        dct = nodes_to_dict(node_dct)
        validate_node(name, node.dict(), network_dict=dct)
        return self.__class__.from_dict(dct, validation=False)

    def convert_nodes(
        self,
        names: list[str] | tuple[str],
        new_node_types: list[type[Node]] | tuple[type[Node]],
        new_node_kwargs: list[dict[str, Any]] | tuple[dict[str, Any]],
    ) -> BayesNet:
        node_dct = dict(self.nodes)
        for i, name in enumerate(names):
            node_dct[name] = new_node_types[i].from_node(
                node_dct[name], **new_node_kwargs[i]
            )

        x = self.copy(update={"nodes": node_dct})
        err = validate_model(self.__class__, x.dict())[2]
        if err:
            raise err
        return x
