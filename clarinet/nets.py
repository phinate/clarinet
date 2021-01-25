__all__ = ["BayesNet", "_BayesNet", "net_from_dict"]

from typing import Dict
from typing import Any

from functools import singledispatchmethod
from dataclasses import asdict
from dataclasses import dataclass

from .nodes import Node, CategoricalNode
from .validation import validate_dict


@dataclass(frozen=True)
class _BayesNet:
    nodes: Dict[str, Node]


@dataclass(frozen=True)
class BayesNet(_BayesNet):
    @singledispatchmethod
    def add_node(self, node) -> _BayesNet:  # type: ignore # noqa
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
        return net_from_dict(dct)


def net_from_dict(network_dict: Dict[str, Dict[str, Any]]) -> BayesNet:
    # validation step
    # TODO: jsonschema
    validate_dict(network_dict)
    nodes: Dict[str, Node] = {}
    for i, items in enumerate(network_dict.items()):
        name, node_dict = items
        if "name" in node_dict.keys():
            del node_dict["name"]
        if "categories" in node_dict.keys():
            nodes[name] = CategoricalNode(name, **node_dict)
        else:
            nodes[name] = Node(name, **node_dict)
        # display_text = node.display_text or name  # TODO daft

    return BayesNet(nodes)
