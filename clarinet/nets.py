__all__ = ["BayesNet"]

from typing import Dict
from typing import Any

from functools import singledispatchmethod
from pydantic import BaseModel

from flax.core import freeze, unfreeze, FrozenDict

from .nodes import Node, CategoricalNode
from .validation import validate_node, validate_model_dict
from .dictutils import nodes_to_dict, modelstring_to_dict


class BayesNet(BaseModel):
    nodes: FrozenDict[str, Node]
    modelstring: str = ""
    # for pydantic

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {FrozenDict: lambda t: unfreeze(t)}

    @classmethod
    def from_dict(  # type: ignore
        cls,
        network_dict: Dict[str, Dict[str, Any]],
        validation: bool = True,
        modelstring: str = ""
    ):  # type: ignore
        # validation step
        # TODO: jsonschema
        if validation:
            validate_model_dict(network_dict)
        nodes: Dict[str, Node] = {}
        for i, items in enumerate(network_dict.items()):
            name, node_dict = items
            # convert to tuples for immutability
            if "parents" in node_dict.keys():
                node_dict["parents"] = tuple(node_dict["parents"])
            if "children" in node_dict.keys():
                node_dict["children"] = tuple(node_dict["children"])
            if "name" in node_dict.keys():
                del node_dict["name"]

            # special casing
            if "categories" in node_dict.keys():
                node_dict["categories"] = tuple(node_dict["categories"])
                nodes[name] = CategoricalNode(name=name, **node_dict)
            else:
                nodes[name] = Node(name=name, **node_dict)
            # display_text = node.display_text or name  # TODO daft

        return cls(freeze(nodes), modelstring)

    @classmethod
    def from_modelstring(cls, modelstring: str):  # type: ignore
        return cls.from_dict(modelstring_to_dict(modelstring))

    @singledispatchmethod
    def add_node(self, node):  # type: ignore
        raise NotImplementedError(
            f"Type '{type(node)}' of node was not recognised")

    @add_node.register
    def _(self, node: Node):  # type: ignore
        node_dct = unfreeze(self.nodes)
        name = node.name
        # refactor for node
        node_dct[name] = node
        # make sure to complete arcs from their other ends
        for child in node.children:
            node_dct[child] = node_dct[child].add_parents([name])
        for parent in node.parents:
            node_dct[parent] = node_dct[parent].add_children([name])
        dct = nodes_to_dict(node_dct)
        validate_node(name, node.dict(), network_dict=dct)
        return self.__class__.from_dict(dct, validation=False)
