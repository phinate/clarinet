import jax.numpy as jnp
import pytest

from clarinet import DiscreteNode
from clarinet import Node

name = "test_name"


@pytest.mark.parametrize(
    ("prob_table", "expected_prob_table"),
    (
        pytest.param([], jnp.array([]), id="empty list"),
        pytest.param(
            [[0.3, 0.7], [0.7, 0.3]],
            jnp.array([[0.3, 0.7], [0.7, 0.3]]),
            id="non-empty list",
        ),
        pytest.param(
            jnp.array([[0.3, 0.7], [0.7, 0.3]]),
            jnp.array([[0.3, 0.7], [0.7, 0.3]]),
            id="non-empty jax array",
        ),
    ),
)
def test_basic_functionality(prob_table, expected_prob_table):
    parents, children = ["A", "B"], ["C", "D"]
    x = DiscreteNode(
        name=name, parents=parents, children=children, prob_table=prob_table
    )
    assert jnp.allclose(x.dict()["prob_table"], expected_prob_table)
    x.json()


def test_prob_table():
    pass  # for when shape checks and sum(probs)=1 checks exist


def test_from_node():
    n = Node(name=name, parents=["F"])  # arbitrary
    prob_table = jnp.array([[0.3, 0.7], [0.7, 0.3]])
    DiscreteNode.from_node(n, prob_table=prob_table)
