import pytest
from pydantic import ValidationError

from clarinet import Node

name = "test_name"


@pytest.mark.parametrize(
    ("parents", "children", "expected"),
    (
        pytest.param(
            [], [], dict(name=name, parents=(), children=()), id="empty lists"
        ),
        pytest.param(
            ["A", "B"],
            ["C", "D"],
            dict(name=name, parents=("A", "B"), children=("C", "D")),
            id="non-empty lists",
        ),
    ),
)
def test_basic_functionality(parents, children, expected):
    x = Node(name=name, parents=parents, children=children)
    assert x.dict() == expected
    x.json()


def test_modify_parents():
    x = Node(name=name, parents=["A"])
    y = x.add_parents(["B"])
    assert set(y.parents) == {"A", "B"}, f"parents don't match {y.parents}"


def test_modify_parents_typeerror():
    with pytest.raises(TypeError):
        x = Node(name=name, parents=["A"])
        x.add_parents(3)  # type: ignore


def test_modify_parents_validerror():
    with pytest.raises(ValidationError):
        Node(name=name, parents=["A"]).add_parents([lambda: 3])  # type: ignore


def test_modify_children():
    x = Node(name=name, children=["A"])
    y = x.add_children(["B"])
    assert set(y.children) == {"A", "B"}, f"children don't match {y.children}"


def test_modify_children_typeerror():
    with pytest.raises(TypeError):
        x = Node(name=name, children=["A"])
        x.add_children(3)  # type: ignore


def test_modify_children_validerror():
    with pytest.raises(ValidationError):
        Node(name=name, children=["A"]).add_children([lambda: 33])  # type: ignore


def test_from_node():
    x = Node(name=name, parents=["B"], children=["A"])
    assert Node.from_node(x) == x
