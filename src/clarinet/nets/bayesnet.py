from __future__ import annotations

__all__ = ["BayesNet"]

from functools import partial, singledispatchmethod
from typing import Any, Dict, List, Optional, Sequence, no_type_check

import numpy as np
from immutables import Map
from pydantic import BaseModel, root_validator, validator
from scipy.sparse import csr_matrix

from ..modelstring import Modelstring
from ..nodes import BaseDiscrete, DiscreteNode, Node


class BayesNet(BaseModel):
    nodes: Map[str, Node]
    link_matrix: csr_matrix
    link_ordering: Map[str, int]
    modelstring: Modelstring = Modelstring("")

    # for pydantic
    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        json_encoders = {
            Map: lambda t: {name: node for name, node in t.items()},
            np.ndarray: lambda t: t.tolist(),
            csr_matrix: lambda t: None,
        }
        fields = {"link_matrix": {"exclude": True}}
        keep_untouched = (singledispatchmethod,)

    def __getitem__(self, item: str) -> Node:
        return self.nodes[item]

    @validator("nodes", pre=True)
    def dict_to_map(cls, dct: Dict[str, Node]) -> Map[str, Node]:
        return Map(dct)

    @staticmethod
    def _validate_prob_table(
        nodes: Dict[str, Dict[str, Any]],
        name: str,
        prob_table: np.ndarray,
        has_parents: bool,
    ) -> None:
        target = nodes[name]
        table = np.array(prob_table)
        num_states = len(target["states"])
        assert (
            table.shape[-1] == num_states
        ), f"{name} should have a probability table with last dimension of size {num_states} ({table.shape[-1]} given)"

        if has_parents:
            parent_states_sizes = [len(nodes[p]["states"]) for p in target["parents"]]
            assert set(parent_states_sizes) == set(
                table.shape[:-1]
            ), f"{name} has incorrect shape, needs to be (*len(parent states), ..., len(states))"

        # by fixing the values of the parent nodes, we define a distribution, so we need to check
        # it sums to unit probability in all cases
        assert np.isclose(
            table.sum(axis=-1).prod(), 1
        ), f"Probability over states of node'{name}' doesn't sum to 1!"  # to account for possible truncation -- needed?

    # this doesn't pick up cycles that occur when searching for node-centric cycles
    # not to worry -- I think this is done easier through the link matrix impl
    @staticmethod
    def _recursive_cycle_check(
        name: str,
        children: Sequence[str],
        network_dict: Dict[str, Dict[str, Any]],
    ) -> None:
        for child in children:
            entry = network_dict[child]
            if "children" in entry.keys():
                if entry["children"] != []:
                    # will show first error based on order of nodes in dict
                    assert (
                        name not in entry["children"]
                    ), f"Network has a cycle -- can cycle back to '{name}'!"
                    BayesNet._recursive_cycle_check(
                        name, entry["children"], network_dict
                    )

    @staticmethod
    def _validate_node(
        name: str, node_dict: Dict[str, Any], network_dict: Dict[str, Dict[str, Any]]
    ) -> None:

        cycle_check = partial(
            BayesNet._recursive_cycle_check, network_dict=network_dict
        )

        has_parents = False
        if "parents" in node_dict.keys():
            if node_dict["parents"] != [] and node_dict["parents"] != ():
                has_parents = True
        has_children = False
        if "children" in node_dict.keys():
            if node_dict["children"] != [] and node_dict["children"] != ():
                has_children = True

        # check for isolated nodes
        #     if not has_parents and not has_children:
        #         in_keys = False
        #         for second_node_dict in network_dict.values():
        #             if "parents" in second_node_dict.keys():
        #                 if name in second_node_dict["parents"]:
        #                     in_keys = True
        #                     break

        #             if "children" in second_node_dict.keys():
        #                 if name in second_node_dict["children"]:
        #                     in_keys = True
        #                     break
        #         assert in_keys, f"Node {name} is isolated!"

        # check validity of declared parents
        if has_parents:
            for parent in node_dict["parents"]:
                assert parent != name, (
                    f"Self-cycle found in '{name}' " + "(not allowed in a DAG!)"
                )
                assert parent in network_dict.keys(), (
                    f"'{parent}' is declared as a parent of '{name}', "
                    + "but not declared as a node."
                )
                assert "children" in network_dict[parent].keys(), (
                    f"{name} declared as child for '{parent}', "
                    + "but '{parent}' has no children."
                )
                assert name in network_dict[parent]["children"], (
                    f"'{name}' must be declared a child of '{parent}', "
                    + f"since '{parent}' is listed as a parent of '{name}'."
                )

        # vice-versa for children
        if has_children:
            children = node_dict["children"]
            for child in children:
                assert child != name, (
                    f"Self-cycle found in '{name}' " + "(not allowed in a DAG!)"
                )
                assert child in network_dict.keys(), (
                    f"'{child}' is declared as a child of '{name}', "
                    + "but not declared as a node."
                )
                assert "parents" in network_dict[child].keys(), (
                    f"'{name}' declared as parent for '{child}', "
                    + "but '{child}' has no parents."
                )
                assert name in network_dict[child]["parents"], (
                    f"'{name}' must be declared a parent of '{child}', "
                    + f"since '{child}' is listed as a child of '{name}'."
                )
            # check recursively to see if any child links back to this node
            cycle_check(name, children)
        if "prob_table" in node_dict.keys() and "states" in node_dict.keys():
            if len(node_dict["prob_table"]):
                BayesNet._validate_prob_table(
                    network_dict, name, node_dict["prob_table"], has_parents
                )

    @staticmethod
    def _nodes_to_dict(nodes: Map[str, Node]) -> Dict[str, Dict[str, Any]]:
        vals = []
        for n in nodes.values():
            vals.append(n.dict())
        return {v["name"]: v for v in vals}

    @validator("nodes", pre=True)
    def validate_model_dict(cls, nodes: Map[str, Node]) -> Map[str, Any]:
        assert len(nodes.keys()) == len(set(nodes.keys())), "Duplicate nodes found!"

        network_dict = cls._nodes_to_dict(nodes)
        node_check = partial(cls._validate_node, network_dict=network_dict)

        for name, node in network_dict.items():
            node_check(name, node)

        return nodes

    @root_validator(skip_on_failure=True, pre=True)
    def init_modelstring(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["modelstring"] == "":
            for node in values["nodes"].values():
                parents = node.parents
                values["modelstring"] += f"[{node.name}"
                values["modelstring"] += f"|{':'.join(parents)}]" if parents else "]"
        return values

    @root_validator(skip_on_failure=True, pre=True)
    def init_link_matrix(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        nodes = values["nodes"].values()

        ordering = {node.name: i for i, node in enumerate(nodes)}

        m = csr_matrix((len(ordering), len(ordering)), dtype=int)

        for i, node in enumerate(nodes):
            for parent in node.parents:
                assert parent in ordering.keys(), f"{parent} is not declared as a node!"
                m[ordering[parent], i] = 1

        values["link_matrix"] = m
        values["link_ordering"] = Map(ordering)

        return values

    @classmethod
    def from_dict(
        cls,
        network_dict: Dict[str, Dict[str, Any]],
        modelstring: str = "",
    ) -> BayesNet:
        nodes: Dict[str, Node] = {}
        for item in network_dict.items():
            name, node_dict = item
            if "name" not in node_dict.keys():
                node_dict["name"] = name
            # special casing
            if "states" in node_dict.keys():
                nodes[name] = DiscreteNode(**node_dict)
            elif "prob_table" in node_dict.keys():
                nodes[name] = BaseDiscrete(**node_dict)
            else:
                nodes[name] = Node(**node_dict)
            # display_text = node.display_text or name  # TODO daft
        return cls(nodes=nodes, modelstring=modelstring)

    @classmethod
    def from_modelstring(cls, modelstring: str) -> BayesNet:
        return cls.from_dict(Modelstring.to_dict(modelstring), modelstring=modelstring)

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
        dct = BayesNet._nodes_to_dict(Map(node_dct))
        BayesNet._validate_node(name, node.dict(), network_dict=dct)
        return self.copy(update={"nodes": node_dct})

    def convert_nodes(
        self,
        names: Sequence[str],
        new_node_types: Sequence[type[Node]],
        new_node_kwargs: Sequence[Dict[str, Any]],
    ) -> BayesNet:
        nodes = dict(self.nodes)
        for i, name in enumerate(names):
            nodes[name] = new_node_types[i].from_node(nodes[name], **new_node_kwargs[i])
        net_dct = BayesNet._nodes_to_dict(Map(nodes))
        for name in names:
            BayesNet._validate_node(name, net_dct[name], net_dct)
        return self.copy(update={"nodes": nodes})

    # TODO: add test
    @no_type_check
    @singledispatchmethod
    def add_prob_tables(
        self,
        names,
        tables: Sequence[Any],
        states: Optional[Sequence[Any]] = None,
    ):
        pass

    @no_type_check
    @add_prob_tables.register
    def single_name(
        self,
        names: str,
        tables: Sequence[Any],
        states: Optional[Sequence[Any]] = None,
    ):
        new_node_types: List[type[Node]] = []

        is_categorical = type(self.nodes[names]) == DiscreteNode
        if states:
            new_node_types.append(DiscreteNode)
        elif is_categorical:
            new_node_types.append(DiscreteNode)
            states = self.nodes[names].states
        else:
            new_node_types.append(BaseDiscrete)
        if states:
            new_node_kwargs = [dict(prob_table=tables, states=states)]
        else:
            new_node_kwargs = [
                dict(
                    prob_table=tables,
                )
            ]
        return self.convert_nodes(
            names, new_node_types=new_node_types, new_node_kwargs=new_node_kwargs
        )

    @no_type_check
    @add_prob_tables.register
    def multi_name(
        self,
        names: list,
        tables: Sequence[Any],
        states: Optional[Sequence[Any]] = None,
    ):
        new_node_types: List[Any] = [None] * len(names)
        if states:
            new_states = list(states)

        for i, name in enumerate(names):
            is_categorical = type(self.nodes[name]) == DiscreteNode

            if is_categorical:
                new_node_types[i] = DiscreteNode
                if states:
                    new_states[i] = self.nodes[name].states
            elif states:
                if new_states[i] is not None:
                    new_node_types[i] = DiscreteNode
                else:
                    new_states[i] = []
                    new_node_types[i] = BaseDiscrete
            else:
                new_node_types[i] = BaseDiscrete
        if states:
            new_node_kwargs = [
                dict(prob_table=table, states=state)
                for table, state in zip(tables, states)
            ]
        else:
            new_node_kwargs = [
                dict(
                    prob_table=table,
                )
                for table in tables
            ]
        return self.convert_nodes(
            names, new_node_types=new_node_types, new_node_kwargs=new_node_kwargs
        )
