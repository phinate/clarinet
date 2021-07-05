from __future__ import annotations

import re
from typing import Any, Dict

modelstring_regex = re.compile(
    r"((\[[\w *]+\|\w+[\w ]*(:\w+[\w ]*)*\])|(\[[\w *]+\]))+", re.IGNORECASE
)


class Modelstring(str):
    @classmethod
    def __get_validators__(cls) -> Any:  # todo: properly type as generator
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> Modelstring:
        if not isinstance(v, str):
            raise TypeError("Modelstring must be a string!")
        m = modelstring_regex.fullmatch(v)
        if not m:
            raise ValueError(
                "Invalid modelstring -- should follow "
                + "bnlearn convention of [Node name|List of parent nodes]"
            )
        return cls(v)

    @staticmethod
    def to_dict(modelstring: str) -> Dict[str, Dict[str, Any]]:
        nodes = modelstring.replace("]", "").split("[")[1:]
        model_dict: dict[str, dict[str, Any]] = {}
        for n in nodes:
            if "|" in n:
                name, parents = n.split("|")
                if ":" in n:
                    parent_list = parents.split(":")
                    model_dict[name] = dict(parents=parent_list, children=[])
                else:
                    model_dict[name] = dict(parents=[parents], children=[])
            else:
                model_dict[n] = dict(parents=[], children=[])

        for name, node_dict in model_dict.items():
            for parent in node_dict["parents"]:
                assert (
                    parent in model_dict.keys()
                ), f"{parent} is declared as a parent but not declared as a node!"
                model_dict[parent]["children"].append(name)
        return model_dict
