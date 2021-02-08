import jax.numpy as jnp
import pytest

from clarinet import DiscreteNode

name = "test_name"


@pytest.mark.parametrize(
    'prob_table',
    (
        pytest.param([], id="empty list"),
        pytest.param([[0.3, 0.7], [0.7, 0.3]], id="non-empty list"),
        pytest.param(
            jnp.array([[0.3, 0.7], [0.7, 0.3]]),
            id="non-empty jax array"
        )

    )
)
def basic_functionality(prob_table):
    parents, children = ["A", "B"], ["C", "D"]
    x = DiscreteNode(
        name=name,
        parents=parents,
        children=children,
        prob_table=prob_table
    )
    x.dict()
    x.json()
