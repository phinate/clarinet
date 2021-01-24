__all__ = ["BayesNet", "_BayesNet", "net_from_dict"]

from typing import Dict
from typing import Any

from functools import singledispatchmethod
from dataclasses import asdict

from flax import struct  # type: ignore

from .nodes import Node
from .validation import validate_dict


@struct.dataclass
class _BayesNet:
    nodes: Dict[str, Node]


@struct.dataclass
class BayesNet(_BayesNet):
    @singledispatchmethod
    def add_node(self, node) -> _BayesNet:
        raise NotImplementedError(
            f"Type '{type(node)}' of node was not recognised")

    @add_node.register
    def _(self, node: Node) -> _BayesNet:
        dct = asdict(self)["nodes"]
        node_dict = asdict(node)
        name = node_dict.pop("name", None)
        dct[name] = node_dict
        # make sure to complete arcs from their other ends
        for child in node.children:
            dct[child]["parents"].append(name)
        for parent in node.parents:
            dct[parent]["children"].append(name)
        return net_from_dict(dct)  # type: ignore  # noqa


def net_from_dict(network_dict: Dict[str, Dict[str, Any]]) -> BayesNet:
    # validation step
    # TODO: jsonschema
    validate_dict(network_dict)
    nodes = {}
    for i, items in enumerate(network_dict.items()):
        name, node_dict = items
        if "name" in node_dict.keys():
            del node_dict["name"]
        if "categories" in node_dict.keys():
            node = CategoricalNode(name, **node_dict)  # type: ignore  # noqa
        else:
            node = Node(name, **node_dict)  # type: ignore  # noqa
        nodes[name] = node
        # display_text = node.display_text or name  # TODO daft

    return BayesNet(nodes)  # type: ignore  # noqa
