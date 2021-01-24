__all__ = ["recursive_cycle_check", "validate_node", "validate_dict"]

from typing import List
from typing import Dict
from typing import Any

from functools import partial


def recursive_cycle_check(
    name: str, children: List[str], network_dict: Dict[str, Dict[str, Any]]
) -> None:
    for child in children:
        entry = network_dict[child]
        if "children" in entry.keys():
            # will show first error based on order of nodes in dict
            assert (
                name not in entry["children"]
            ), f"Network has a cycle -- can cycle back to '{name}'!"
            recursive_cycle_check(name, entry["children"], network_dict)


def validate_node(
    name: str,
    node_dict: Dict[str, Any],
    network_dict: Dict[str, Dict[str, Any]]
) -> None:

    cycle_check = partial(
        recursive_cycle_check, network_dict=network_dict
    )

    has_parents = False
    if "parents" in node_dict.keys():
        if node_dict["parents"] != []:
            has_parents = True
    has_children = False
    if "children" in node_dict.keys():
        if node_dict["children"] != []:
            has_children = True
    # print(name, has_parents,has_children)

    # check for isolated nodes
    if not has_parents and not has_children:

        in_keys = False
        for second_node_dict in network_dict.values():
            if "parents" in second_node_dict.keys():
                if name in second_node_dict["parents"]:
                    in_keys = True
                    break

            if "children" in second_node_dict.keys():
                if name in second_node_dict["children"]:
                    in_keys = True
                    break
        assert in_keys, f"Node {name} is isolated!"

    # check validity of declared parents
    if has_parents:
        for parent in node_dict["parents"]:
            assert (
                parent != name
            ), (f"Self-cycle found in '{name}' "
                + "(not allowed in a DAG!)")
            assert (
                parent in network_dict.keys()
            ), (f"'{parent}' is declared as a parent of '{name}', "
                + "but not declared as a node.")
            assert (
                "children" in network_dict[parent].keys()
            ), (f"{name} declared as child for '{parent}', "
                + "but '{parent}' has no children.")
            assert (
                name in network_dict[parent]["children"]
            ), (f"'{name}' must be declared a child of '{parent}', "
                + f"since '{parent}' is listed as a parent of '{name}'.")

    # vice-versa for children
    if has_children:
        children = node_dict["children"]
        for child in children:
            assert (
                child != name
            ), (f"Self-cycle found in '{name}' " +
                "(not allowed in a DAG!)")
            assert (
                child in network_dict.keys()
            ), (f"'{child}' is declared as a child of '{name}', " +
                "but not declared as a node.")
            assert (
                "parents" in network_dict[child].keys()
            ), (f"'{name}' declared as parent for '{child}', " +
                "but '{child}' has no parents.")
            assert (
                name in network_dict[child]["parents"]
            ), (f"'{name}' must be declared a parent of '{child}', " +
                f"since '{child}' is listed as a child of '{name}'.")
        # check recursively to see if any child links back to this node
        cycle_check(name, children)


def validate_dict(network_dict: Dict[str, Dict[str, Any]]) -> None:
    assert len(network_dict.keys()) == len(
        set(network_dict.keys())
    ), "Duplicate nodes found!"

    node_check = partial(validate_node, network_dict=network_dict)

    for name, node_dict in network_dict.items():
        node_check(name, node_dict)
