__all__ = ["BayesNet", "_BayesNet", "net_from_dict", "nodes_as_dict"]

from typing import Dict
from typing import Any

from functools import singledispatchmethod
from dataclasses import asdict
from dataclasses import dataclass

from flax.core import freeze, unfreeze, FrozenDict

from .nodes import Node, CategoricalNode, DiscreteNode
from .validation import validate_dict, validate_node


@dataclass(frozen=True)
class _BayesNet:
    nodes: FrozenDict[str, Node]


@dataclass(frozen=True)
class BayesNet(_BayesNet):

    @singledispatchmethod
    def add_node(self, node) -> _BayesNet:  # type: ignore # noqa
        raise NotImplementedError(
            f"Type '{type(node)}' of node was not recognised")

    @add_node.register
    def _(self, node: Node) -> _BayesNet:
        node_dct = unfreeze(self.nodes)
        name = node.name
        # refactor for node
        node_dct[name] = node
        # make sure to complete arcs from their other ends
        for child in node.children:
            node_dct[child].set_parents([name], keep_current=True)
        for parent in node.parents:
            node_dct[parent].set_parents([name], keep_current=True)
        dct = nodes_as_dict(node_dct)
        validate_node(name, asdict(node), network_dict=dct)
        return net_from_dict(dct, validation=False)

    def as_dict(
        self, serializable: bool = False
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        net_copy = unfreeze(self.nodes)
        if serializable:  # TODO: investigate flax serialization of frozendict
            for name, node in net_copy.items():
                if isinstance(node, DiscreteNode):
                    net_copy[name] = node.make_serializable()
            return {'nodes': nodes_as_dict(net_copy)}
        else:
            return {'nodes': nodes_as_dict(net_copy)}


def net_from_dict(
    network_dict: Dict[str, Dict[str, Any]],
    validation: bool = True
) -> BayesNet:
    # validation step
    # TODO: jsonschema
    if validation:
        validate_dict(network_dict)
    nodes: Dict[str, Node] = {}
    for i, items in enumerate(network_dict.items()):
        name, node_dict = items

        # convert to tuples for immutability
        if "parents" in node_dict.keys():
            node_dict["parents"] = tuple(node_dict["parents"])
        if "parents" in node_dict.keys():
            node_dict["parents"] = tuple(node_dict["parents"])
        if "name" in node_dict.keys():
            del node_dict["name"]

        # special casing
        if "categories" in node_dict.keys():
            node_dict["categories"] = tuple(node_dict["categories"])
            nodes[name] = CategoricalNode(name, **node_dict)
        else:
            nodes[name] = Node(name, **node_dict)
        # display_text = node.display_text or name  # TODO daft

    return BayesNet(freeze(nodes))


def nodes_as_dict(nodes: Dict[str, Node]) -> Dict[str, Dict[str, Any]]:
    vals = list(map(asdict, nodes.values()))
    return {v['name']: v for v in vals}
