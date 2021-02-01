__all__ = ["BayesNet", "_BayesNet", "_BayesNetDefaults"]

from typing import Dict
from typing import Any

from functools import singledispatchmethod
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

from flax.core import freeze, unfreeze, FrozenDict

from .nodes import Node, CategoricalNode, DiscreteNode
from .validation import validate_node, validate_model_dict
from .dictutils import nodes_to_dict, modelstring_to_dict


@dataclass(frozen=True)
class _BayesNet:
    nodes: FrozenDict[str, Node]


@dataclass(frozen=True)
class _BayesNetDefaults:
    modelstring: str = field(default="")


@dataclass(frozen=True)
class BayesNet(_BayesNetDefaults, _BayesNet):

    @classmethod
    def from_dict(
        cls,
        network_dict: Dict[str, Dict[str, Any]],
        validation: bool = True,
        modelstring: str = ""
    ) -> _BayesNet:
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
                nodes[name] = CategoricalNode(name, **node_dict)
            else:
                nodes[name] = Node(name, **node_dict)
            # display_text = node.display_text or name  # TODO daft

        return cls(freeze(nodes), modelstring)

    @classmethod
    def from_modelstring(cls, modelstring: str) -> _BayesNet:
        return cls.from_dict(modelstring_to_dict(modelstring))

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
            node_dct[child] = node_dct[child].set_parents(
                [name],
                keep_current=True
            )
        for parent in node.parents:
            node_dct[parent] = node_dct[parent].set_children(
                [name],
                keep_current=True
            )
        dct = nodes_to_dict(node_dct)
        validate_node(name, asdict(node), network_dict=dct)
        return BayesNet.from_dict(dct, validation=False)

    def as_dict(
        self, serializable: bool = False
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        net_copy = unfreeze(self.nodes)
        if serializable:  # TODO: investigate flax serialization of frozendict
            for name, node in net_copy.items():
                if isinstance(node, DiscreteNode):
                    net_copy[name] = node.make_serializable()
            return {'nodes': nodes_to_dict(net_copy)}
        else:
            return {'nodes': nodes_to_dict(net_copy)}
