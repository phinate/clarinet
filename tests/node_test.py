import pytest

from clarinet import Node

name = "test_name"


@pytest.mark.parametrize(
    ('parents', 'children'),
    (
        pytest.param([], [], id="empty lists"),
        pytest.param(["A", "B"], ["C", "D"], id="non-empty lists")
    )
)
def basic_functionality(parents, children):
    x = Node(name=name, parents=parents, children=children)
    x.dict()
    x.json()


def modify_parents():
    x = Node(name, parents=["A"])
    y = x.add_parents(["B"])
    assert y.parents == ("A", "B"), f"parents don't match {y.parents}"


def modify_children():
    x = Node(name, children=["A"])
    y = x.add_children(["B"])
    assert y.children == ("A", "B"), f"children don't match {y.children}"
