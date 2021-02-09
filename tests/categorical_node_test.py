import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from clarinet import CategoricalNode

name = "test_name"
prob_table = jnp.array([[0.3, 0.7], [0.7, 0.3]])


def test_basic_functionality():
    categories = ["c1", "c2"]
    parents, children = ["A", "B"], ["C", "D"]
    x = CategoricalNode(
        name=name,
        parents=parents,
        children=children,
        prob_table=prob_table,
        categories=categories
    )
    assert x.categories == ("c1", "c2")
    x.json()


@pytest.mark.parametrize(
    'categories',
    (
        pytest.param([], id="empty"),
        pytest.param(["e"], id="only one category"),
        pytest.param("fagfdsgfd", id="string")
    )
)
def test_categories_validerror(categories):
    with pytest.raises(ValidationError):
        CategoricalNode(name=name, categories=categories)
