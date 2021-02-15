from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from .validation import validate_modelstring

__all__ = ["nodes_to_dict", "modelstring_to_dict"]


if TYPE_CHECKING:
    from .nodes import Node


def nodes_to_dict(nodes: dict[str, Node]) -> dict[str, dict[str, Any]]:
    vals = []
    for n in nodes.values():
        vals.append(n.dict())
    return {v['name']: v for v in vals}


def modelstring_to_dict(modelstring: str) -> dict[str, dict[str, Any]]:
    validate_modelstring(modelstring)
    nodes = modelstring.replace(']', "").split("[")[1:]
    model_dict: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if '|' in n:
            name, parents = n.split('|')
            if ':' in n:
                parent_list = parents.split(':')
                model_dict[name] = dict(parents=parent_list, children=[])
            else:
                model_dict[name] = dict(parents=[parents], children=[])
        else:
            model_dict[n] = dict(parents=[], children=[])

    for name, node_dict in model_dict.items():
        for parent in node_dict['parents']:
            model_dict[parent]['children'].append(name)
    return model_dict
